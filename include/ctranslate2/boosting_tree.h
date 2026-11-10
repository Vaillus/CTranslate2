#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoding_utils.h"

namespace ctranslate2 {

  // A prefix tree of keyterm token sequences with failure links, ported from
  // icefall's ContextGraph (icefall/context_graph.py): same build, same walk
  // (forward_one_step in strict mode), same withdrawal (finalize).
  //
  // A tree position is a node index; 0 is the root. The tree is never changed
  // after it is built: a keyterm update builds a new tree.
  class BoostingTree {
  public:
    // One boosting weight per token sequence. Sequences sharing a node keep the
    // largest weight on it, as in ContextGraph.build.
    BoostingTree(const std::vector<std::vector<size_t>>& token_sequences,
                 const std::vector<float>& weights);

    // Number of nodes, the root excluded (ContextGraph.num_nodes).
    size_t num_nodes() const {
      return _nodes.size() - 1;
    }

    // ContextGraph.forward_one_step(state, token) in strict mode: returns the
    // next tree position and sets the score change of `token` from `state`.
    int32_t step(int32_t state, size_t token, float& score) const;

    // The tree position reached from the root after `tokens`.
    int32_t walk(const int32_t* tokens, size_t num_tokens) const;

    // The row of score changes from `state`: `default_score` is the score change
    // of every token except the listed ones, `listed` holds the tokens with a
    // value of their own (the children of `state` and of the nodes on its
    // failure-link chain). A token that is in no node, the end-of-transcript
    // token included, gets `default_score`, which is ContextGraph.finalize.
    void score_row(int32_t state,
                   float& default_score,
                   std::vector<std::pair<size_t, float>>& listed) const;

  private:
    struct Node {
      float token_score = 0;
      float node_score = 0;
      float output_score = 0;
      bool is_end = false;
      int32_t fail = 0;
      std::unordered_map<size_t, int32_t> next;
    };

    int32_t child(int32_t node, size_t token) const;
    void fill_fail_output();

    std::vector<Node> _nodes;
  };


  // The logits processor that walks a BoostingTree over each hypothesis and adds
  // its row of score changes. In beam search it is applied at the second hook,
  // on the log-probabilities; in greedy decoding and sampling, on the logits.
  class BoostingTreeProcessor : public LogitsProcessor {
  public:
    BoostingTreeProcessor(std::shared_ptr<const BoostingTree> tree);

    bool after_log_softmax() const override {
      return true;
    }

    void apply(dim_t step,
               StorageView& scores,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const std::shared_ptr<const BoostingTree> _tree;
  };

}
