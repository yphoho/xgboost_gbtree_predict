#ifndef _XGBOOST_GBTREE_MODEL_H_
#define _XGBOOST_GBTREE_MODEL_H_

#include <stdio.h>
#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


namespace regression {

typedef uint64_t bst_ulong;
typedef float bst_float;
typedef bst_float TSplitCond;


struct GBTreeModelParam {
    /*! \brief number of trees */
    int num_trees;
    /*! \brief number of roots */
    int num_roots;
    /*! \brief number of features to be used by trees */
    int num_feature;
    /*! \brief pad this space, for backward compatibility reason.*/
    int pad_32bit;
    /*! \brief deprecated padding space. */
    int64_t num_pbuffer_deprecated;
    /*!
     * \brief how many output group a single instance can produce
     *  this affects the behavior of number of output we have:
     *    suppose we have n instance and k group, output will be k * n
     */
    int num_output_group;
    /*! \brief size of leaf vector needed in tree */
    int size_leaf_vector;
    /*! \brief reserved parameters */
    int reserved[32];
};


/*! \brief meta parameters of the tree */
struct TreeParam {
    /*! \brief number of start root */
    int num_roots;
    /*! \brief total number of nodes */
    int num_nodes;
    /*!\brief number of deleted nodes */
    int num_deleted;
    /*! \brief maximum depth, this is a statistics of the tree */
    int max_depth;
    /*! \brief number of features used for tree construction */
    int num_feature;
    /*!
     * \brief leaf vector size, used for vector tree
     * used to store more than one dimensional information in tree
     */
    int size_leaf_vector;
    /*! \brief reserved part, make sure alignment works for 64bit */
    int reserved[31];
};


class Node {
  public:
    /*! \brief index of left child */
    inline int cleft() const {
        return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright() const {
        return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault() const {
        return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index() const {
        return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left() const {
        return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool is_leaf() const {
        return cleft_ == -1;
    }
    /*! \return get split condition of the node */
    inline TSplitCond split_cond() const {
        return (this->info_).split_cond;
    }

  private:
    union Info{
        bst_float leaf_value;
        TSplitCond split_cond;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int parent_;
    // pointer to left, right
    int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    unsigned sindex_;
    // extra info
    Info info_;
};


class GBTree {
  public:
    void load(const char *model_file) {
        FILE *fp = fopen(model_file, "rb");

        // LearnerModelParam
        fseek(fp, sizeof_learn_param, SEEK_CUR);

        uint64_t len;

        // name_obj_
        fread(&len, sizeof(len), 1, fp);
        fseek(fp, len, SEEK_CUR);

        // name_gbm_
        fread(&len, sizeof(len), 1, fp);
        fseek(fp, len, SEEK_CUR);

        // GBTreeModelParam
        GBTreeModelParam gbtree_model_param;
        fread(&gbtree_model_param, sizeof(gbtree_model_param), 1, fp);

        tree_.resize(gbtree_model_param.num_trees);

        for(size_t i = 0; i < tree_.size(); i++) {
            // TreeParam
            TreeParam tree_param;
            fread(&tree_param, sizeof(tree_param), 1, fp);

            tree_[i].resize(tree_param.num_nodes);

            // Node
            for(int j = 0; j < tree_param.num_nodes; j++) {
                fread(&tree_[i][j], sizeof(Node), 1, fp);
            }

            // NodeStat
            fseek(fp, sizeof_node_stat * tree_param.num_nodes, SEEK_CUR);

            // leaf_vector
            if(tree_param.size_leaf_vector != 0) {
                fread(&len, sizeof(len), 1, fp);
                fseek(fp, len, SEEK_CUR);
            }

        }
        // tree_info * num_trees

        // other info

        fclose(fp);

        // dump();
    }

    void predict(const std::map<int, float> &feat, std::vector<int> &out_preds) {
        out_preds.resize(tree_.size());
        for(size_t i = 0; i < tree_.size(); i++) {
            int pid = 0;
            while(!tree_[i][pid].is_leaf()) {
                unsigned split_index = tree_[i][pid].split_index();
                if((feat.find(split_index) == feat.end())) {
                    pid = GetNext(i, pid, 0, true);
                } else {
                    pid = GetNext(i, pid, feat.find(split_index)->second, false);
                }
                // std::cout << i << ": " << pid << std::endl;
            }
            out_preds[i] = pid;
        }
    }

    void dump() {
        for(size_t i = 0; i < tree_.size(); i++) {
            for(size_t j = 0; j < tree_[i].size(); j++) {
                std::cout << i << " " << j << ": "
                    << tree_[i][j].split_index() << ", "
                    << tree_[i][j].split_cond() << ", "
                    << tree_[i][j].cleft() << ", "
                    << tree_[i][j].cright() << ", "
                    << tree_[i][j].cdefault() << std::endl;
            }
            std::cout << std::endl;
        }
    }

  private:
    /*! \brief get next position of the tree given current pid */
    int GetNext(int tree_num, int pid, bst_float fvalue, bool is_unknown) const {
        bst_float split_value = tree_[tree_num][pid].split_cond();
        if (is_unknown) {
            return tree_[tree_num][pid].cdefault();
        } else {
            if (fvalue < split_value) {
                return tree_[tree_num][pid].cleft();
            } else {
                return tree_[tree_num][pid].cright();
            }
        }
    }

  private:
    static const int sizeof_learn_param = 136;
    // static const int sizeof_gbtree_model_param = 160;
    // static const int sizeof_tree_param = 148;
    static const int sizeof_node_stat = 16;

    std::vector< std::vector<Node> > tree_;
};


// int main(int argc, char *argv[]) {
//     const char *model_file = argv[1];
//
//     GBTree gbtree;
//     gbtree.load(model_file);
//
//     const char *test_file = argv[2];
//
//     FILE *fp = fopen(test_file, "rb");
//
//     char line[1024];
//     bool print_it = false;
//     while(fgets(line, 1024, fp) != 0) {
//         line[strlen(line) - 1] = 0;  // trim '\n'
//
//         std::vector<std::string> terms;
//         boost::split(terms, line, boost::is_any_of(" "));
//
//         std::map<int, float> feat;
//         for(size_t i = 1; i < terms.size(); i++) {
//             std::vector<std::string> kv;
//             boost::split(kv, terms[i], boost::is_any_of(":"));
//
//             int k = boost::lexical_cast<int>(kv[0]);
//             float v = boost::lexical_cast<float>(kv[1]);
//             feat.insert(std::make_pair(k, v));
//         }
//
//         if(print_it) {
//             print_it = false;
//             for(std::map<int, float>::const_iterator it = feat.begin(); it != feat.end(); it++) {
//                 std::cout << it->first << ":" << it->second << " ";
//             }
//             std::cout << std::endl;
//         }
//
//         std::vector<int> leaf_index;
//         gbtree.predict(feat, leaf_index);
//
//         for(size_t i = 0; i < leaf_index.size(); i++) {
//             std::cout << leaf_index[i] << " ";
//         }
//         std::cout << std::endl;
//     }
//
//     fclose(fp);
//
//     return 0;
// }

}

#endif
