#pragma once

/*!
* file distributed_wordembedding.h
* \brief Class Distributed_wordembedding describles the main frame of Distributed WordEmbedding and some useful functions 
*/

#include <vector>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <sys/stat.h>

#include "util.h"
#include "multiverso.h"
#include "huffman_encoder.h"
#include "data_block.h"
#include "parameter_loader.h"
#include "trainer.h"
#include "reader.h"
#include "log.h"
#include "constant.h"

namespace multiverso
{
    namespace wordembedding
    {
        extern std::string g_log_suffix;
        class Trainer;
        class Distributed_wordembedding
        {
        public:
            Distributed_wordembedding(){}
            /*!
            * \brief Run Function contains everything
            */
            void Run(int argc, char *argv[]);

        private:
            clock_t start_;
            int process_id_;
            Option* option_;
            Dictionary* dictionary_;
            HuffmanEncoder* huffman_encoder_;
            Sampler* sampler_;
            Reader* reader_;
            std::thread collect_wordcount_thread_;
            bool is_running_;
            std::vector<Trainer*> trainers_;
            WordEmbedding *WordEmbedding_;
            /*!
            * \brief Create a new thread which is used for 
            * \calculating the speed of word processing.
            */
            void StartThread();
            void StartCollectWordcountThread(
                std::vector<multiverso::TrainerBase*> &trainer, WordEmbedding *WordEmbedding);
            void StopCollectWordcountThread();
            /*!
            * \brief Read the word from the train_file
            * \param word word saved by string
            * \param fin   train_filename
            */
            bool ReadWord(char *word, FILE *fin);
            /*!
            * \brief Load Dictionary from the vocabulary_file
            * \param opt Some model-set setparams 
            * \param dictionary save the vocabulary and its frequency
            * \param huffman_encoder convert dictionary to the huffman_code
            */
            int64 LoadVocab(Option *opt, Dictionary *dictionary,
                HuffmanEncoder *huffman_encoder);
            /*!
            * \brief Get the file total wordnumber
            */
            int64 GetFileSize(const char *filename);
            /*!
            * \brief Complete the train task with multiverso
            */
            void Train(int argc, char *argv[]);
            void TrainNeuralNetwork();
            /*!
            * \brief Create a new table in the multiverso
            */
            void CreateMultiversoParameterTable(multiverso::integer_t table_id,
                multiverso::integer_t rows, multiverso::integer_t cols,
                multiverso::Type type, multiverso::Format default_format);
            /*!
            * \brief Push the datablock into the multiverso and datablock_queue
            */
            void PushDataBlock(std::queue<DataBlock*> &datablock_queue,
                DataBlock* data_block);
            /*!
            * \brief Prepare parameter table in the multiverso
            */
            void PrepareMultiversoParameterTables(Option *opt,
                Dictionary *dictionary);
            /*!
            * \brief Loaddata from train_file to datablock
            * \param datablock the datablock which needs to be assigned
            * \param reader some useful function for calling
            * \param size datablock limit byte size
            */
            void LoadData(DataBlock *data_block, Reader *reader, int64 size);
            /*!
            * \brief Remove datablock which is finished by multiverso thread
            * \param datablock_queue store the pushed datablocks
            */
            void RemoveDoneDataBlock(std::queue<DataBlock*> &datablock_queue);
            // No copying allowed
            Distributed_wordembedding(const Distributed_wordembedding&);
            void operator=(const Distributed_wordembedding&);
        };
    }
}
