#ifndef DISTRIBUTED_WORD_EMBEDDING_DISTRIBUTED_WORDEMBEDDING_H_
#define DISTRIBUTED_WORD_EMBEDDING_DISTRIBUTED_WORDEMBEDDING_H_

#pragma once

/*!
* file distributed_wordembedding.h
* \brief Class Distributed_wordembedding describles the main frame of Distributed WordEmbedding and some useful functions
*/

#include <vector>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <omp.h>

#include "multiverso/multiverso.h"
#include "multiverso/table/matrix_table.h"
#include "multiverso/util/async_buffer.h"

#include "util.h"
#include "huffman_encoder.h"
#include "reader.h"
#include "constant.h"
#include "data_block.h"
#include "trainer.h"
#include "memory_manager.h"
#include "block_queue.h"

namespace multiverso
{
	namespace wordembedding
	{
		extern std::string g_log_suffix;
		class Trainer;
		class WordEmbedding;
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
			bool is_running_;
			WordEmbedding* WordEmbedding_;
			MemoryManager* memory_mamanger_;
			BlockQueue *block_queue_;
			std::thread load_data_thread_;

			MatrixWorkerTable<float>* worker_input_table_;
			MatrixWorkerTable<float>* worker_output_table_;
			MatrixServerTable<float>* server_input_table_;
			MatrixServerTable<float>* server_output_table_;

			MatrixWorkerTable<float>* worker_input_gradient_table_;
			MatrixWorkerTable<float>* worker_output_gradient_table_;
			MatrixServerTable<float>* server_input_gradient_table_;
			MatrixServerTable<float>* server_output_gradient_table_;

			/*!
			* \brief Load Dictionary from the vocabulary_file
			* \param opt Some model-set setparams
			* \param dictionary save the vocabulary and its frequency
			* \param huffman_encoder convert dictionary to the huffman_code
			*/
			int64 LoadVocab(Option *opt, Dictionary *dictionary,
				HuffmanEncoder *huffman_encoder);

			/*!
			* \brief Loaddata from train_file to datablock
			* \param datablock the datablock which needs to be assigned
			* \param reader some useful function for calling
			* \param size datablock limit byte size
			*/
			//void LoadData(DataBlock *data_block, Reader *reader, int64 size);

			/*!
			* \brief Complete the train task with multiverso
			*/
			void Train(int argc, char *argv[]);
			HRESULT TrainNeuralNetwork();

			void PrepareData(DataBlock *data_block);

			void RequestParameter(DataBlock *data_block);

			void AddDeltaParameter(DataBlock *data_block);

			void AddRows(MatrixWorkerTable<float>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size);

			void AddRow(MatrixWorkerTable<float>* table_, int row_id, real* ptr, int size);

			void GetRow(MatrixWorkerTable<float>* table_, int row_id, real* ptr, int size);
			
			void GetRows(MatrixWorkerTable<float>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size);
			/*!
			* \brief Prepare parameter table in the multiverso
			*/
			void PrepareParameterTables(Option *opt,
				Dictionary *dictionary);

			void StartLoadDataThread(BlockQueue *block_queue,
				Reader *reader, int64 file_size);

			void LoadOneBlock(DataBlock *data_block,
				Reader *reader, int64 size);

			DataBlock* GetDataFromQueue(BlockQueue *block_queue);

			DataBlock* GetBlockAndPrepareParameter(BlockQueue *block_queue_);
			/*!
			* \brief Save the input-embedding vectors in file_path
			* \param file_path
			* \param is_binary, the format of file
			* 1 - save the vectors in the binary format,
			* 2 - save the vectors in the ascii format
			*/
			void SaveEmbedding(const char *file_path, bool is_binary);
		};
	}
}
#endif