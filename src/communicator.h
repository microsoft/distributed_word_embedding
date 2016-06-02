#ifndef DISTRIBUTED_WORD_EMBEDDING_COMMUNICATOR_H_
#define DISTRIBUTED_WORD_EMBEDDING_COMMUNICATOR_H_

#include "multiverso/table/matrix_table.h"
#include "multiverso/table/kv_table.h"
#include "multiverso/updater/updater.h"

#include "memory_manager.h"
#include "block_queue.h"

namespace multiverso
{
	namespace wordembedding
	{

		class Communicator
		{

		public:
			Communicator(Option* option);
			~Communicator();

			void RequestBlocks(int size, std::vector<real*> &blocks);
			void ReturnBlocks(std::vector<real*> &blocks);

			void RequestParameter(DataBlock *data_block);

			void AddDeltaParameter(DataBlock *data_block);

			int64 GetWordCount();
			void AddWordCount(int word_count_num);

			void GetWorkerTableRows(std::vector<int> row_nums, std::vector<real*> &blocks, int embeding_size);

			void PrepareParameterTables(int row_size, int column_size);

		private:
			Option* option_ = nullptr;
			MemoryManager* memory_mamanger_ = nullptr;
			int process_id_;

			MatrixWorkerTable<real>* worker_input_table_ = nullptr;
			MatrixWorkerTable<real>* worker_output_table_ = nullptr;
			MatrixServerTable<real>* server_input_table_ = nullptr;
			MatrixServerTable<real>* server_output_table_ = nullptr;

			MatrixWorkerTable<real>* worker_input_gradient_table_ = nullptr;
			MatrixWorkerTable<real>* worker_output_gradient_table_ = nullptr;
			MatrixServerTable<real>* server_input_gradient_table_ = nullptr;
			MatrixServerTable<real>* server_output_gradient_table_ = nullptr;

			KVWorkerTable<int, int64>* worker_wordcount_table_ = nullptr;
			KVServerTable<int, int64>* server_wordcount_table_ = nullptr;

			void ClearParameterTables();

			void GetRows(MatrixWorkerTable<real>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size);
			void RequestParameterByTableId(DataBlock *data_block, int table_id, std::vector<int> &nodes, std::vector<real*> &blocks);
			void SetDataBlockEmbedding(DataBlock *data_block, std::vector<real*> &blocks, std::vector<int> &nodes, int table_id);

			void AddRows(MatrixWorkerTable<real>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size);
			void AddParameterByTableId(DataBlock *data_block, int table_id, std::vector<int> &nodes, std::vector<real*> &blocks, std::vector<real*> &recycle_blocks);
			void GetDeltaLoop(DataBlock *data_block, std::vector<real*> &blocks, std::vector<int> &nodes, int table_id, std::vector<real*> &recycle_blocks);
		};
	}
}
#endif