#include "ctranslate2/boosting_tree.h"

#include <algorithm>
#include <deque>
#include <stdexcept>

#include "ctranslate2/primitives.h"
#include "dispatch.h"

namespace ctranslate2 {

  static constexpr int32_t root = 0;

  BoostingTree::BoostingTree(const std::vector<std::vector<size_t>>& token_sequences,
                             const std::vector<float>& weights)
    : _nodes(1)
  {
    if (weights.size() != token_sequences.size())
      throw std::invalid_argument("BoostingTree: expected one weight per token sequence, got "
                                  + std::to_string(weights.size()) + " weights for "
                                  + std::to_string(token_sequences.size()) + " sequences");

    // The trie, as ContextGraph.build makes it.
    for (size_t s = 0; s < token_sequences.size(); ++s) {
      const auto& tokens = token_sequences[s];
      const float weight = weights[s];
      int32_t node = root;

      for (size_t i = 0; i < tokens.size(); ++i) {
        const size_t token = tokens[i];
        const bool last = (i + 1 == tokens.size());
        int32_t next = child(node, token);

        if (next < 0) {
          next = static_cast<int32_t>(_nodes.size());
          Node created;
          created.token_score = weight;
          created.node_score = _nodes[node].node_score + weight;
          created.output_score = last ? created.node_score : 0;
          created.is_end = last;
          _nodes.emplace_back(std::move(created));
          _nodes[node].next.emplace(token, next);
        } else {
          // A shared node keeps the largest weight.
          Node& shared = _nodes[next];
          shared.token_score = std::max(weight, shared.token_score);
          shared.node_score = _nodes[node].node_score + shared.token_score;
          shared.is_end = last || shared.is_end;
          shared.output_score = shared.is_end ? shared.node_score : 0;
        }

        node = next;
      }
    }

    fill_fail_output();
  }

  int32_t BoostingTree::child(int32_t node, size_t token) const {
    const auto& next = _nodes[node].next;
    const auto it = next.find(token);
    return it == next.end() ? -1 : it->second;
  }

  // Failure links and output scores, by one breadth-first pass, as
  // ContextGraph._fill_fail_output.
  void BoostingTree::fill_fail_output() {
    std::deque<int32_t> queue;
    for (const auto& [token, node] : _nodes[root].next) {
      _nodes[node].fail = root;
      queue.push_back(node);
    }

    while (!queue.empty()) {
      const int32_t current = queue.front();
      queue.pop_front();

      for (const auto& [token, node] : _nodes[current].next) {
        int32_t fail = _nodes[current].fail;
        int32_t next = child(fail, token);
        if (next >= 0) {
          fail = next;
        } else {
          fail = _nodes[fail].fail;
          while (child(fail, token) < 0) {
            fail = _nodes[fail].fail;
            if (fail == root)
              break;
          }
          next = child(fail, token);
          if (next >= 0)
            fail = next;
        }
        _nodes[node].fail = fail;

        // The output: the nearest end node on the failure-link chain.
        int32_t output = _nodes[node].fail;
        while (!_nodes[output].is_end) {
          output = _nodes[output].fail;
          if (output == root) {
            output = -1;
            break;
          }
        }
        if (output >= 0)
          _nodes[node].output_score += _nodes[output].output_score;

        queue.push_back(node);
      }
    }
  }

  int32_t BoostingTree::step(int32_t state, size_t token, float& score) const {
    int32_t node = child(state, token);

    if (node >= 0) {
      score = _nodes[node].token_score;
    } else {
      // Follow the failure links until a node has `token` as a child, or the root.
      node = _nodes[state].fail;
      while (child(node, token) < 0) {
        node = _nodes[node].fail;
        if (node == root)
          break;
      }
      const int32_t next = child(node, token);
      if (next >= 0)
        node = next;

      // The withdrawal of what the hypothesis received since the matched node.
      score = _nodes[node].node_score - _nodes[state].node_score;
    }

    score += _nodes[node].output_score;
    return node;
  }

  int32_t BoostingTree::walk(const int32_t* tokens, size_t num_tokens) const {
    int32_t state = root;
    float score = 0;
    for (size_t i = 0; i < num_tokens; ++i)
      state = step(state, static_cast<size_t>(tokens[i]), score);
    return state;
  }

  void BoostingTree::score_row(int32_t state,
                               float& default_score,
                               std::vector<std::pair<size_t, float>>& listed) const {
    // A token that is a child of no node on the chain falls back to the root.
    default_score = _nodes[root].node_score - _nodes[state].node_score + _nodes[root].output_score;
    listed.clear();

    // The candidates: the children of `state` and of each node on its
    // failure-link chain, down to the root's children. Each one is scored by
    // step(), so the row is forward_one_step token by token.
    std::vector<size_t> candidates;
    int32_t node = state;
    while (true) {
      for (const auto& [token, next] : _nodes[node].next)
        candidates.push_back(token);
      if (node == root)
        break;
      node = _nodes[node].fail;
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    listed.reserve(candidates.size());
    for (const size_t token : candidates) {
      float score = 0;
      step(state, token, score);
      listed.emplace_back(token, score);
    }
  }


  BoostingTreeProcessor::BoostingTreeProcessor(std::shared_ptr<const BoostingTree> tree)
    : _tree(std::move(tree))
  {
  }

  void BoostingTreeProcessor::apply(dim_t,
                                    StorageView& scores,
                                    DisableTokens&,
                                    const StorageView& sequences,
                                    const std::vector<dim_t>& batch_offset,
                                    const std::vector<std::vector<size_t>>* prefix) {
    const dim_t num_rows = scores.dim(0);
    const dim_t vocabulary_size = scores.dim(-1);
    const dim_t length = sequences ? sequences.dim(-1) : 0;

    std::vector<float> row_defaults(num_rows);
    std::vector<int32_t> flat_indices;
    std::vector<float> values;
    std::vector<std::pair<size_t, float>> listed;
    bool any_default = false;

    for (dim_t row = 0; row < num_rows; ++row) {
      // The tree position is recomputed from the tokens the hypothesis wrote
      // since its prompt: beam search reorders and duplicates hypotheses
      // between steps, so a position stored per row would follow the wrong one.
      int32_t state = root;
      if (length > 0) {
        const dim_t begin = get_sample_begin(num_rows, row, batch_offset, prefix);
        if (begin < length)
          state = _tree->walk(sequences.index<int32_t>({row, begin}), length - begin);
      }

      float default_score = 0;
      _tree->score_row(state, default_score, listed);
      row_defaults[row] = default_score;
      any_default = any_default || default_score != 0;

      for (const auto& [token, score] : listed) {
        if (static_cast<dim_t>(token) >= vocabulary_size)
          continue;
        const float delta = score - default_score;
        if (delta == 0)
          continue;
        flat_indices.push_back(static_cast<int32_t>(row * vocabulary_size + token));
        values.push_back(delta);
      }
    }

    const Device device = scores.device();
    const DataType dtype = scores.dtype();

    // One value for every token of a row, then the listed tokens their own.
    if (any_default) {
      const StorageView defaults = StorageView({num_rows}, row_defaults).to(dtype).to(device);
      DEVICE_AND_TYPE_DISPATCH(device, dtype,
                               primitives<D>::add_depth_broadcast(defaults.data<T>(),
                                                                  scores.data<T>(),
                                                                  num_rows,
                                                                  scores.size()));
    }

    if (!flat_indices.empty()) {
      const dim_t num_indices = flat_indices.size();
      const StorageView indices({num_indices}, flat_indices, device);
      const StorageView deltas = StorageView({num_indices}, values).to(dtype).to(device);
      DEVICE_AND_TYPE_DISPATCH(device, dtype,
                               primitives<D>::indexed_add(scores.data<T>(),
                                                          deltas.data<T>(),
                                                          indices.data<int32_t>(),
                                                          num_indices));
    }
  }

}
