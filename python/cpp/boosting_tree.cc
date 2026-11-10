#include "module.h"

#include <pybind11/stl.h>

#include <ctranslate2/boosting_tree.h>

namespace ctranslate2 {
  namespace python {

    void register_boosting_tree(py::module& m) {
      py::class_<BoostingTree, std::shared_ptr<BoostingTree>>(
        m, "BoostingTree",
        R"pbdoc(
            A prefix tree of keyterm token sequences with failure links, built once
            in C++ and passed to :meth:`ctranslate2.models.Whisper.generate`, which
            only reads it. The build, the walk and the withdrawal are those of
            icefall's ContextGraph (strict mode). The tree is never changed: to
            change the keyterms, build a new tree.
        )pbdoc")

        .def(py::init<const std::vector<std::vector<size_t>>&, const std::vector<float>&>(),
             py::arg("token_sequences"),
             py::arg("weights"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Builds the tree.

                 Arguments:
                   token_sequences: One token ID sequence per entry.
                   weights: The boosting weight of each sequence, added per token.
             )pbdoc")

        .def_property_readonly("num_nodes", &BoostingTree::num_nodes,
                               "Number of nodes, the root excluded.")

        .def("score_changes",
             [](const BoostingTree& tree, const std::vector<int32_t>& tokens) {
               const int32_t state = tree.walk(tokens.data(), tokens.size());
               float default_score = 0;
               std::vector<std::pair<size_t, float>> listed;
               tree.score_row(state, default_score, listed);
               std::unordered_map<size_t, float> own(listed.begin(), listed.end());
               return std::make_pair(default_score, own);
             },
             py::arg("tokens"),
             R"pbdoc(
                 The row of score changes after ``tokens``, walked from the root.

                 Returns:
                   A tuple ``(value, listed)``: ``value`` is the score change of every
                   token not in ``listed``, the end-of-transcript token included, and
                   ``listed`` maps the other tokens to their own score change.
             )pbdoc")
        ;
    }

  }
}
