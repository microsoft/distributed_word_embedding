#pragma once

/*!
* file trainer.h
* \brief Class Trainer trains the model by every trainiteration
*/

#include <thread>
#include <chrono>

#include "multiverso.h"
#include "data_block.h"
#include "constant.h"
#include "util.h"
#include "huffman_encoder.h"
#include "word_embedding.h"
#include "memory_manager.h"
#include "barrier.h"

namespace multiverso
{
    namespace wordembedding
    {
        class WordEmbedding;
        extern std::string g_log_suffix;
        class Trainer : public multiverso::TrainerBase
        {
        public:
            int64 word_count;
            Trainer(int trainer_id, Option *option, Barrier* barrier,
                Dictionary* dictionary, WordEmbedding* WordEmbedding,
                MemoryManager* memory_mamanger);
            /*!
            * /brief Train one datablock
            */
            void TrainIteration(multiverso::DataBlockBase* data_block) override;

        private:
            int process_count_;
            int process_id_;
            int trainer_id_;
            Option *option_;
            real *hidden_act_, *hidden_err_;
            WordEmbedding* WordEmbedding_;
            multiverso::Barrier *barrier_;
            Dictionary* dictionary_;
            MemoryManager* memory_mamanger_;
            int train_count_;
            clock_t start_, now_;
            FILE* log_file_;

            /*!
            * \brief Save the input-embedding vectors in file_path
            * \param file_path
            * \param is_binary, the format of file
            * 1 - save the vectors in the binary format,
            * 2 - save the vectors in the ascii format
            */
            void SaveEmbedding(const char *file_path, bool is_binary);
            /*!
            * \brief Copy the needed parameter from buffer to blocks
            */
            void CopyRow(real* ptr, multiverso::Row<real>& row, int size);
            void CopyParameter(std::vector<int>& input_nodes,
                std::vector<int>& output_nodes);
            /*!
            * \brief Add delta to the parameter stored in the 
            * \buffer and send it to multiverso
            */
            void AddRow(real* ptr, int table_id,
                int row_id, int size);
            void AddDeltaParameter(std::vector<int>& input_nodes,
                std::vector<int>& output_nodes);

            //No copying allowed
            Trainer(const Trainer&);
            void operator=(const Trainer&);
        };
    }
}
