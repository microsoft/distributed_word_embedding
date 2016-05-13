#include "distributed_wordembedding.h"

namespace multiverso
{
	namespace wordembedding
	{
		template <typename T>
		void filler(std::vector<T> &v){
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<float> dis(-1.0, 1.0);

			for (int i = 0; i<v.size(); i++)
			{
				v[i] = dis(gen);
			}
		}

		void Distributed_wordembedding::PrepareParameterTables(Option *opt, Dictionary *dictionary){
			worker_input_table_ = new MatrixWorkerTable<float>(dictionary->Size(), opt->embeding_size);
			worker_output_table_ = new MatrixWorkerTable<float>(dictionary->Size(), opt->embeding_size);
			server_input_table_ = new MatrixServerTable<float>(dictionary->Size(), opt->embeding_size, &filler);
			server_output_table_ = new MatrixServerTable<float>(dictionary->Size(), opt->embeding_size, &filler);

			if (option_->use_adagrad){
				worker_input_gradient_table_ = new MatrixWorkerTable<float>(dictionary->Size(), opt->embeding_size);
				worker_output_gradient_table_ = new MatrixWorkerTable<float>(dictionary->Size(), opt->embeding_size);
				server_input_gradient_table_ = new MatrixServerTable<float>(dictionary->Size(), opt->embeding_size);
				server_output_gradient_table_ = new MatrixServerTable<float>(dictionary->Size(), opt->embeding_size);
			}
		}

		void Distributed_wordembedding::LoadOneBlock(DataBlock *data_block,
			Reader *reader, int64 size)
		{
			clock_t start = clock();
			
			//multiverso::Log::Info("Beigin Load One Block.\n");

			data_block->ClearSentences();
			reader->ResetSize(size);
			while (true)
			{
				int64 word_count = 0;
				int *sentence = new (std::nothrow)int[kMaxSentenceLength + 2];
				assert(sentence != nullptr);
				int sentence_length = reader->GetSentence(sentence, word_count);
				if (sentence_length > 0)
				{
					data_block->AddSentence(sentence, sentence_length,
						word_count, (uint64)rand() * 10000 + (uint64)rand());
				}
				else
				{
					//Reader read eof or has read data_block->size bytes before,
					//reader_->GetSentence will return 0
					delete[] sentence;
					break;
				}
			}

			multiverso::Log::Info("LoadOneDataBlockTime:%lfs\n",
				(clock() - start) / (double)CLOCKS_PER_SEC);
		}

		void Distributed_wordembedding::StartLoadDataThread(BlockQueue *block_queue, Reader *reader, int64 file_size){
			int data_block_count = 0;
			for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch)
			{
				clock_t start_epoch = clock();
				reader_->ResetStart();
				for (int64 cur = 0; cur < file_size; cur += option_->data_block_size)
				{
					DataBlock *data_block = new (std::nothrow)DataBlock();
					assert(data_block!=nullptr);
					LoadOneBlock(data_block, reader, option_->data_block_size);

					multiverso::Log::Info("Load the %d Data Block\n", data_block_count);
					data_block_count++;

					std::unique_lock<std::mutex> lock(block_queue->mtx);
					(block_queue->queues).push(data_block);
					(block_queue->repo_not_empty).notify_all();
					lock.unlock();
				}
			}
		}

		DataBlock* Distributed_wordembedding::GetDataFromQueue(BlockQueue *block_queue){
			std::unique_lock<std::mutex> lock(block_queue->mtx);
			// item buffer is empty, just wait here.
			while (block_queue->queues.size() == 0) {
				multiverso::Log::Info("Waiting For Loading Data Block...\n");
				(block_queue->repo_not_empty).wait(lock);
			}

			DataBlock *temp = block_queue->queues.front();
			multiverso::Log::Info("Geting Data Block From Queue...\n");
			block_queue->queues.pop();
			lock.unlock();
			return temp;
		}

		DataBlock* Distributed_wordembedding::GetBlockAndPrepareParameter(BlockQueue *block_queue_){
			DataBlock* data_block = GetDataFromQueue(block_queue_);
			if (data_block->Size() == 0){
				return nullptr;
			}
			data_block->MallocMemory(dictionary_->Size(),option_);
			PrepareData(data_block);
			RequestParameter(data_block);
			return data_block;
		}

		
		HRESULT Distributed_wordembedding::TrainNeuralNetwork(){
			int64 file_size = GetFileSize(option_->train_file);
			multiverso::Log::Info("train-file-size:%lld, data_block_size:%lld\n",
				file_size, option_->data_block_size);

			block_queue_ = new BlockQueue();
			memory_mamanger_ = new MemoryManager(option_->embeding_size);

			std::vector<Trainer*> trainers;
			WordEmbedding_ = new WordEmbedding(option_, huffman_encoder_,
				sampler_, dictionary_->Size());
			assert(WordEmbedding_!= nullptr);

			for (int i = 0; i < option_->thread_cnt; ++i)
			{
				trainers.push_back(new (std::nothrow) Trainer(i, option_, dictionary_, WordEmbedding_, memory_mamanger_));
				assert(trainers[i] != nullptr);
			}

			load_data_thread_ = std::thread(&Distributed_wordembedding::StartLoadDataThread, this, block_queue_, reader_, file_size);
			load_data_thread_.detach();
			//load_data_thread_.join();

			int data_block_count = 0;
			DataBlock *next_block = nullptr;
			DataBlock *data_block = nullptr;

			data_block = GetBlockAndPrepareParameter(block_queue_);
			if (data_block == nullptr){
				multiverso::Log::Info("Please Change the Bigger Block Size.\n");
				return E_INVALIDARG;
			}
			data_block_count++;

			int64 all = file_size / option_->data_block_size + 1;
			for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch)
			{
				clock_t start_epoch = clock();
				for (int64 cur = 0; cur < all; cur ++)
				{
					clock_t start_block = clock();
					bool flag = (cur != (all - 1) || (cur_epoch != (option_->epoch - 1)));
					if (option_->is_pipeline == false){
						if (data_block != nullptr){
							multiverso::Log::Info("Get the %d Data Block and Request done.\n", data_block_count);
							data_block_count++;

							#pragma omp parallel for num_threads(option_->thread_cnt)
							for (int i = 0; i < option_->thread_cnt; ++i){
								trainers[i]->TrainIteration(data_block);
							}

							AddDeltaParameter(data_block);
							delete data_block;
						}
						if (flag){
							data_block = GetBlockAndPrepareParameter(block_queue_);
						}
					}
					else{
						bool flag = (cur != (all - 1) || (cur_epoch != (option_->epoch - 1)));
						if (data_block != nullptr){
							#pragma omp parallel num_threads(option_->thread_cnt+1)
							{
								if (omp_get_thread_num() == option_->thread_cnt){
									if (flag){
										next_block = GetBlockAndPrepareParameter(block_queue_);
									}
								}
								else{
									//for (int i = 0; i < option_->thread_cnt; ++i){
									trainers[omp_get_thread_num()]->TrainIteration(data_block);
									//}
								}
							}

							AddDeltaParameter(data_block);
							delete data_block;
							//(if next_block == nullptr) then data_block is null,we not run next block
							data_block = next_block;
							next_block = nullptr;
						}
					}

					multiverso::Log::Info("Dealing one block time:%lfs\n",
						(clock() - start_block) / (double)CLOCKS_PER_SEC);
				}

				multiverso::Log::Info("Dealing one epoch time:%lfs\n",
					(clock() - start_epoch) / (double)CLOCKS_PER_SEC);
			}

			multiverso::Log::Info("Rank %d Pushed %d datablocks\n",
				process_id_, data_block_count);

			SaveEmbedding(option_->output_file, option_->output_binary);

			if (option_->is_pipeline == true){
				delete next_block;
			}
			delete memory_mamanger_;
			delete WordEmbedding_;
			delete block_queue_;
			for (auto trainer : trainers)
			{
				delete trainer;
			}
		}

		void Distributed_wordembedding::SaveEmbedding(const char *file_path, bool is_binary)
		{
			FILE* fid = nullptr;
			real* row = new real[option_->embeding_size]();
			clock_t start = clock();
			if (is_binary)
			{
				fid = fopen(file_path, "wb");
				fprintf(fid, "%d %d\n", dictionary_->Size(), option_->embeding_size);
				for (int i = 0; i < dictionary_->Size(); ++i)
				{
					fprintf(fid, "%s ",
						dictionary_->GetWordInfo(i)->word.c_str());
					worker_input_table_->Get(i, row, option_->embeding_size);

					for (int j = 0; j < option_->embeding_size; ++j)
					{
						real tmp = row[j];
						fwrite(&tmp, sizeof(real), 1, fid);
					}

					fprintf(fid, "\n");
				}

				fclose(fid);
			}
			else
			{
				fid = fopen(file_path, "wt");
				fprintf(fid, "%d %d\n", dictionary_->Size(), option_->embeding_size);

				for (int i = 0; i < dictionary_->Size(); ++i)
				{
					fprintf(fid, "%s ", dictionary_->GetWordInfo(i)->word.c_str());
					worker_input_table_->Get(i, row, option_->embeding_size);

					for (int j = 0; j < option_->embeding_size; ++j)
						fprintf(fid, "%lf ", row[j]);

					fprintf(fid, "\n");
				}

				fclose(fid);
			}

			multiverso::Log::Info("Saving Embedding time:%lfs\n",
				(clock() - start) / (double)CLOCKS_PER_SEC);
			delete[]row;
		}

		void Distributed_wordembedding::PrepareData(DataBlock *data_block){
			clock_t start = clock();
			WordEmbedding_->PrepareData(data_block);
			multiverso::Log::Info("Prepare data time:%lfs\n",
				(clock() - start) / (double)CLOCKS_PER_SEC);
			multiverso::Log::Info("Rank %d Prepare data done\n", multiverso::MV_Rank());
		}

		void Distributed_wordembedding::AddRows(MatrixWorkerTable<float>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size){
			AddOption option;
			table_->Add(row_ids, ptrs, size, &option);
		}

		void Distributed_wordembedding::GetRows(MatrixWorkerTable<float>* table_, std::vector<int> row_ids, std::vector<real *> ptrs, int size){
			table_->Get(row_ids, ptrs, size);
		}

		void Distributed_wordembedding::RequestParameter(DataBlock *data_block)
		{
			clock_t start = clock();

			std::vector<int> input_nodes(data_block->input_nodes.begin(), data_block->input_nodes.end());
			std::vector<int> output_nodes(data_block->output_nodes.begin(), data_block->output_nodes.end());

			std::vector<real*> input_blocks;
			std::vector<real*> output_blocks;

			//Request blocks to store parameters
			memory_mamanger_->RequestBlocks(input_nodes.size(), input_blocks);
			memory_mamanger_->RequestBlocks(output_nodes.size(), output_blocks);
			assert(input_blocks.size() == input_nodes.size());
			assert(output_blocks.size() == output_nodes.size());

			GetRows(worker_input_table_, input_nodes, input_blocks, option_->embeding_size);
			GetRows(worker_output_table_, output_nodes, output_blocks, option_->embeding_size);

			for (int i = 0; i < input_nodes.size(); ++i)
			{
				data_block->SetWeightIE(input_nodes[i], input_blocks[i]);
			}

			for (int i = 0; i < output_nodes.size(); ++i)
			{
				data_block->SetWeightEO(output_nodes[i], output_blocks[i]);
			}

			if (option_->use_adagrad)
			{
				std::vector<real*> input_gradient_blocks;
				std::vector<real*> output_gradient_blocks;

				memory_mamanger_->RequestBlocks(input_nodes.size(), input_gradient_blocks);
				memory_mamanger_->RequestBlocks(output_nodes.size(), output_gradient_blocks);

				GetRows(worker_input_gradient_table_, input_nodes, input_gradient_blocks, option_->embeding_size);
				GetRows(worker_output_gradient_table_, output_nodes, output_gradient_blocks, option_->embeding_size);

				//Copy input-embedding sum of squarsh of gradient 
				for (int i = 0; i < input_nodes.size(); ++i)
				{
					data_block->SetSumGradient2IE(input_nodes[i], input_gradient_blocks[i]);
				}

				//Copy embedding-output sum of squarsh of gradient 
				for (int i = 0; i < output_nodes.size(); ++i)
				{
					data_block->SetSumGradient2EO(output_nodes[i], output_gradient_blocks[i]);
				}
			}

			multiverso::Log::Info("Request Parameters time:%lfs\n",
				(clock() - start) / (double)CLOCKS_PER_SEC);
			multiverso::Log::Info("Rank %d Request Parameter done\n", process_id_);
		}

		//Add delta to local buffer and send it to the parameter sever
		void Distributed_wordembedding::AddDeltaParameter(DataBlock *data_block)
		{
			clock_t start = clock();

			std::vector<int> input_nodes(data_block->input_nodes.begin(), data_block->input_nodes.end());
			std::vector<int> output_nodes(data_block->output_nodes.begin(), data_block->output_nodes.end());
			std::vector<real*> input_blocks;
			std::vector<real*> output_blocks;
			//Request blocks to store parameters
			memory_mamanger_->RequestBlocks(input_nodes.size(), input_blocks);
			memory_mamanger_->RequestBlocks(output_nodes.size(), output_blocks);
			assert(input_blocks.size() == input_nodes.size());
			assert(output_blocks.size() == output_nodes.size());

			GetRows(worker_input_table_, input_nodes, input_blocks, option_->embeding_size);
			GetRows(worker_output_table_, output_nodes, output_blocks, option_->embeding_size);

			for (int i = 0; i < input_nodes.size(); ++i)
			{
				real* new_row = data_block->GetWeightIE(input_nodes[i]);
				real* old_row = input_blocks[i];
				assert(new_row != nullptr);

				//#### to do : if delta < eps
				for (int j = 0; j < option_->embeding_size; ++j)
				{
					old_row[j] = (new_row[j] - old_row[j]) / option_->thread_cnt;
				}
			}

			for (int i = 0; i < output_nodes.size(); ++i)
			{
				real* new_row = data_block->GetWeightEO(output_nodes[i]);
				real* old_row = output_blocks[i];
				assert(new_row != nullptr);

				//#### to do : if delta < eps
				for (int j = 0; j < option_->embeding_size; ++j)
				{
					old_row[j] = (new_row[j] - old_row[j]) / option_->thread_cnt;
				}
			}

			AddRows(worker_input_table_, input_nodes, input_blocks, option_->embeding_size);
			AddRows(worker_output_table_, output_nodes, output_blocks, option_->embeding_size);

			memory_mamanger_->ReturnBlocks(input_blocks);
			memory_mamanger_->ReturnBlocks(output_blocks);

			if (option_->use_adagrad){
				std::vector<real*> input_gradient_blocks;
				std::vector<real*> output_gradient_blocks;

				memory_mamanger_->RequestBlocks(input_nodes.size(), input_gradient_blocks);
				memory_mamanger_->RequestBlocks(output_nodes.size(), output_gradient_blocks);

				GetRows(worker_input_gradient_table_, input_nodes, input_gradient_blocks, option_->embeding_size);
				GetRows(worker_output_gradient_table_, output_nodes, output_gradient_blocks, option_->embeding_size);

				for (int i = 0; i < input_nodes.size(); ++i)
				{
					real* new_row = data_block->GetSumGradient2IE(input_nodes[i]);
					real* old_row = input_gradient_blocks[i];
					assert(new_row != nullptr);

					for (int j = 0; j < option_->embeding_size; ++j)
					{
						old_row[j] = (new_row[j] - old_row[j]) / option_->thread_cnt;
					}
				}

				for (int i = 0; i < output_nodes.size(); ++i)
				{
					real* new_row = data_block->GetSumGradient2EO(output_nodes[i]);
					real* old_row = output_gradient_blocks[i];
					assert(new_row != nullptr);

					//#### to do : if delta < eps
					for (int j = 0; j < option_->embeding_size; ++j)
					{
						old_row[j] = (new_row[j] - old_row[j]) / option_->thread_cnt;
					}
				}

				AddRows(worker_input_gradient_table_, input_nodes, input_gradient_blocks, option_->embeding_size);
				AddRows(worker_output_gradient_table_, output_nodes, output_gradient_blocks, option_->embeding_size);

				memory_mamanger_->ReturnBlocks(input_gradient_blocks);
				memory_mamanger_->ReturnBlocks(output_gradient_blocks);
			}

			multiverso::Log::Info("Add Parameters time:%lfs\n",(clock() - start) / (double)CLOCKS_PER_SEC);
			multiverso::Log::Info("Rank %d Add Parameter done\n", process_id_);
		}

		void Distributed_wordembedding::Train(int argc, char *argv[])
		{
			//####fix later
			argc = 1;
			argv = nullptr;

			multiverso::MV_Init(&argc,argv);
			multiverso::Log::Info("MV Init done.\n");

			MV_Barrier();
			multiverso::Log::Info("MV Barrier done.\n");

			//Mark the node machine number
			process_id_ = multiverso::MV_Rank();

			//create worker table and server table
			PrepareParameterTables(option_, dictionary_);

			//start to train
			HRESULT hr = TrainNeuralNetwork();
			if (FAILED(hr))
			{
				multiverso::Log::Info("TrainNeuralNetwork Result is failed.\n");
			}

			MV_ShutDown();
			multiverso::Log::Info("MV ShutDone done.");
		}

		void Distributed_wordembedding::Run(int argc, char *argv[])
		{
			g_log_suffix = GetSystemTime();
			srand(static_cast<unsigned int>(time(NULL)));
			
			option_ = new (std::nothrow)Option();
			assert(option_ != nullptr);
			
			dictionary_ = new (std::nothrow)Dictionary();
			assert(dictionary_ != nullptr);
			
			huffman_encoder_ = new (std::nothrow)HuffmanEncoder();
			assert(huffman_encoder_ != nullptr);
			//Parse argument and store them in option

			if (argc <= 1)
			{
				option_->PrintUsage();
				return;
			}

			option_->ParseArgs(argc, argv);

			//Read the vocabulary file; create the dictionary
			//and huffman_encoder according opt
			if ((option_->hs == 1) && (option_->negative_num != 0))
			{
				multiverso::Log::Fatal("The Hierarchical Softmax and Negative Sampling is indefinite!\n");
				exit(0);
			}

			multiverso::Log::Info("Loading vocabulary ...\n");
			option_->total_words = LoadVocab(option_, dictionary_,
				huffman_encoder_);
			multiverso::Log::Info("Loaded vocabulary\n");
			//dictionary_->PrintVocab();

			option_->PrintArgs();

			sampler_ = new (std::nothrow)Sampler();
			assert(sampler_ != nullptr);
			if (option_->negative_num)
				sampler_->SetNegativeSamplingDistribution(dictionary_);

			char *filename = new (std::nothrow)char[strlen(option_->train_file) + 1];
			assert(filename != nullptr);
			strcpy(filename, option_->train_file);
			reader_ = new (std::nothrow)Reader(dictionary_, option_, sampler_, filename);
			assert(reader_ != nullptr);
			//Train with multiverso
			this->Train(argc, argv);

			delete option_;
			delete dictionary_;
			delete huffman_encoder_;
			delete sampler_;
			delete reader_;
		}
		 
		//Read the vocabulary file; create the dictionary
		//and huffman_encoder according opt
		int64 Distributed_wordembedding::LoadVocab(Option *opt,
			Dictionary *dictionary, HuffmanEncoder *huffman_encoder)
		{
			int64 total_words = 0;
			char word[kMaxString];
			FILE* fid = nullptr;
			clock_t start = clock();
			multiverso::Log::Info("vocab_file %s\n", opt->read_vocab_file);
			
			if (opt->read_vocab_file != nullptr && strlen(opt->read_vocab_file) > 0)
			{
				multiverso::Log::Info("Begin to load vocabulary file [%s] ...\n",
					opt->read_vocab_file);
				fid = fopen(opt->read_vocab_file, "r");
				if (fid == nullptr)
				{
					multiverso::Log::Fatal("Open vocab_file failed!\n");
					exit(1);
				}
				int word_freq;
				while (fscanf(fid, "%s %d", word, &word_freq) != EOF)
				{
					dictionary->Insert(word, word_freq);
				}
			}

			dictionary->RemoveWordsLessThan(opt->min_count);
			multiverso::Log::Info("Dictionary size: %d\n", dictionary->Size());
			
			total_words = 0;
			for (int i = 0; i < dictionary->Size(); ++i)
				total_words += dictionary->GetWordInfo(i)->freq;
			multiverso::Log::Info("Words in Dictionary %I64d\n", total_words);
			
			multiverso::Log::Info("Loading vocab time:%lfs\n",
				(clock() - start) / (double)CLOCKS_PER_SEC);

			if (opt->hs)
				huffman_encoder->BuildFromTermFrequency(dictionary);
			if (fid != nullptr)
				fclose(fid);

			return total_words;
		}
	}
}
