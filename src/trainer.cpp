#include "trainer.h"
namespace multiverso
{
    namespace wordembedding
    {
        Trainer::Trainer(int trainer_id, Option *option,
            multiverso::Barrier *barrier,
            Dictionary* dictionary, WordEmbedding* WordEmbedding,
            MemoryManager* memory_mamanger)
        {
            trainer_id_ = trainer_id;
            option_ = option;   
            word_count = 0;
            WordEmbedding_ = WordEmbedding;
            barrier_ = barrier;
            dictionary_ = dictionary;
            memory_mamanger_ = memory_mamanger;
            hidden_act_ = (real *)calloc(option_->embeding_size, sizeof(real));
            hidden_err_ = (real *)calloc(option_->embeding_size, sizeof(real));
            process_count_ = -1;
            process_id_ = -1;   
            assert(hidden_act_ != nullptr);
            assert(hidden_err_ != nullptr);
            start_ = 0;
            train_count_ = 0;
            if (trainer_id_ == 0)
            {
                //The log which recordes the begin and end time of TrainIteration()
                char log_name[100];
                sprintf_s(log_name, "trainer%s.txt", g_log_suffix.c_str());
                fopen_s(&log_file_, log_name, "w");
            }
        }


        void Trainer::TrainIteration(multiverso::DataBlockBase *data_block)
        {
            if (process_id_ == -1)
                process_id_ = multiverso::Multiverso::ProcessRank();

            if (trainer_id_ == 0)
                //Record the starting time of the Trainiteration  
                fprintf_s(log_file_, "%lf\n", (clock()) / (double)CLOCKS_PER_SEC);

            multiverso::Log::Info("Rank %d Train %d Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            ++train_count_;
            //Compute the total number of processes
            if (process_count_ == -1)
                process_count_ = multiverso::Multiverso::TotalProcessCount();
            //Get the input_nodes and output_nodes from data_block
            //The input_nodes and output_nodes are stored by ParameterLoader
            DataBlock *data = reinterpret_cast<DataBlock*>(data_block);
            std::vector<int>& input_nodes = data->input_nodes;
            std::vector<int>& output_nodes = data->output_nodes;
            //A trainer only copy or add apart of parameters
            //This trainer should copy or add the parameters according to
            //local_input_nodes and local_output_nodes 
            std::vector<int> local_input_nodes;
            std::vector<int> local_output_nodes;
            for (int i = trainer_id_; i < input_nodes.size(); i += option_->thread_cnt)
                local_input_nodes.push_back(input_nodes[i]);
            for (int i = trainer_id_; i < output_nodes.size(); i += option_->thread_cnt)
                local_output_nodes.push_back(output_nodes[i]);

            if (trainer_id_ == 0)
            {
                multiverso::Log::Info("Rank %d input_size=%d, output_size=%d\n",
                    process_id_, input_nodes.size(), output_nodes.size());
            }

            //Step 1, Copy the parameter from multiverso to WordEmbedding_
            //One trainer only copy a part of parameters
            multiverso::Log::Debug("Rank %d Train %d Copyparameter Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
			
            CopyParameter(local_input_nodes, local_output_nodes);
            if (trainer_id_ == 0)
            {
                multiverso::Row<int64> &copy_row = GetRow<int64>(kWordCountActualTableId, 0);
                WordEmbedding_->word_count_actual = copy_row.At(0);
                WordEmbedding_->UpdateLearningRate();
            }
            multiverso::Log::Debug("Rank %d Train %d Copyparameter end TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            //Wait for all the trainers to finish copying parameter
            barrier_->Wait();
		
            //Step 2, After finishing copying parameter,
            //Use WordEmbedding_ to train a part of data_block
            int64 last_word_count = word_count;
            clock_t start = clock();
            multiverso::Log::Debug("Rank %d Train %d TrainNN Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            WordEmbedding_->Train(data, trainer_id_, option_->thread_cnt,
                word_count, hidden_act_, hidden_err_);
            if (word_count > last_word_count)
            {
                multiverso::Log::Info("TrainNNSpeed: Words/thread/second %lfk\n",
                    ((double)word_count - last_word_count) / 
                    (clock() - start) * (double)CLOCKS_PER_SEC / 1000);
            }
            multiverso::Log::Debug("Rank %d Train %d TrainNN end TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            //Wait for all the trainers to finish training
            barrier_->Wait();
            multiverso::Log::Debug("Rank %d Train %d AddDeltaParameter Begin TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            //Step 3, After finishing training, add the delta of parameters to multiverso
            AddDeltaParameter(local_input_nodes, local_output_nodes);
            if (trainer_id_ == 0)
            {
                multiverso::Row<int64> &copy_row = GetRow<int64>(kWordCountActualTableId, 0);
                Add<int64>(kWordCountActualTableId, 0, 0, WordEmbedding_->word_count_actual - copy_row.At(0));
            }
            multiverso::Log::Debug("Rank %d Train %d AddDeltaParameter end TrainIteration%d ...\n",
                process_id_, trainer_id_, train_count_);
            //If the data_block is the last one,Dump the input-embedding weights 
            if (data->Type() == DataBlockType::Test && trainer_id_ == 0)
            {
                SaveEmbedding(option_->output_file, option_->output_binary);
            }
            //Dump the input-embedding to compute the accuracy
            //Record the accuracy of each dumped-result which is from a Test::type datablock
            if (data->Type() == DataBlockType::Test && process_id_ == 0 && trainer_id_ == 0)    
            {
                SaveEmbedding("tmp.bin", 1);
                char s[128] = { 0 };
                sprintf_s(s, "check.py tmp.bin %d >> records.txt", clock() / CLOCKS_PER_SEC);
                system(s);
                multiverso::Log::Info("The dumped-result has been tested");
            }

            if (trainer_id_ == 0)
            {
                fprintf_s(log_file_, "%lf\n",
                    (clock()) / (double)CLOCKS_PER_SEC);
                fflush(log_file_);
            }
        }

        void Trainer::CopyRow(real* ptr, multiverso::Row<real>& row, int size)
        {
            for (int i = 0; i < size; ++i)
                ptr[i] = row.At(i);
        }


        void Trainer::CopyParameter(std::vector<int>& input_nodes,
            std::vector<int>& output_nodes)
        {
            //Compute the number of necessary memory blocks to store parameter
            std::vector<real*> blocks;
            int current_block = 0;
            size_t total_blocks = (input_nodes.size() + output_nodes.size());
            if (option_->use_adagrad)
                total_blocks *= 2;

            //Request blocks to store parameters
            memory_mamanger_->RequestBlocks(total_blocks, blocks);
            assert(blocks.size() == total_blocks);
            if (blocks.size() != total_blocks)
            {
                multiverso::Log::Error("Rank %d Trainer %d Error to requestBlocks to CopyParameter, allocated_blocks_num=%lld, needed_blocks_num=%lld\n",
                    multiverso::Multiverso::ProcessRank(), trainer_id_, blocks.size(), total_blocks);
                return;
            }

            //Copy input-embedding weights from multiverso to WordEmbedding
            for (int i = 0; i < input_nodes.size(); ++i)
            {
                real* ptr = blocks[current_block++];
                assert(ptr != nullptr);
                CopyRow(ptr, GetRow<real>(kInputEmbeddingTableId,
                    input_nodes[i]), option_->embeding_size);

                WordEmbedding_->SetWeightIE(input_nodes[i], ptr);
            }

            //Copy embedding-output weights from multiverso to WordEmbedding
            for (int i = 0; i < output_nodes.size(); ++i)
            {
                real* ptr = blocks[current_block++];
                assert(ptr != nullptr);
                CopyRow(ptr, GetRow<real>(kEmbeddingOutputTableId,
                    output_nodes[i]), option_->embeding_size);

                WordEmbedding_->SetWeightEO(output_nodes[i], ptr);
            }

            if (option_->use_adagrad)
            {
                //Copy input-embedding sum of squarsh of gradient 
                for (int i = 0; i < input_nodes.size(); ++i)
                {
                    real* ptr = blocks[current_block++];
                    assert(ptr != nullptr);
                    CopyRow(ptr, GetRow<real>(kSumGradient2IETableId,
                        input_nodes[i]), option_->embeding_size);

                    WordEmbedding_->SetSumGradient2IE(input_nodes[i], ptr);
                }

                //Copy embedding-output sum of squarsh of gradient 
                for (int i = 0; i < output_nodes.size(); ++i)
                {
                    real* ptr = blocks[current_block++];
                    assert(ptr != nullptr);
                    CopyRow(ptr, GetRow<real>(kSumGradient2EOTableId,
                        output_nodes[i]), option_->embeding_size);

                    WordEmbedding_->SetSumGradient2EO(output_nodes[i], ptr);
                }
            }
        }


        void Trainer::AddRow(real* ptr, int table_id, int row_id, int size)
        {
            multiverso::Row<real>& row = GetRow<real>(table_id, row_id);
            for (int i = 0; i < size; ++i)
            {
                real delta = (ptr[i] - row.At(i)) / process_count_;
                if (fabs(delta) > kEps)
                    Add<real>(table_id, row_id, i, delta);
            }
        }

        //Add delta to local buffer and send it to the parameter sever
        void Trainer::AddDeltaParameter(std::vector<int>& input_nodes,
            std::vector<int>& output_nodes)
        {
            std::vector<real*> blocks;
            for (int i = 0; i < input_nodes.size(); ++i)
            {
                real* ptr = WordEmbedding_->GetWeightIE(input_nodes[i]);
                assert(ptr != nullptr);
                AddRow(ptr, kInputEmbeddingTableId, input_nodes[i],
                    option_->embeding_size);

                blocks.push_back(ptr);
            }

            for (int i = 0; i < output_nodes.size(); ++i)
            {
                real* ptr = WordEmbedding_->GetWeightEO(output_nodes[i]);
                assert(ptr != nullptr);
                AddRow(ptr, kEmbeddingOutputTableId, output_nodes[i],
                    option_->embeding_size);
                blocks.push_back(ptr);
            }

            if (option_->use_adagrad)
            {
                for (int i = 0; i < input_nodes.size(); ++i)
                {
                    real* ptr = WordEmbedding_->GetSumGradient2IE(input_nodes[i]);
                    assert(ptr != nullptr);
                    AddRow(ptr, kSumGradient2IETableId, input_nodes[i],
                        option_->embeding_size);
                    blocks.push_back(ptr);
                }

                for (int i = 0; i < output_nodes.size(); ++i)
                {
                    real* ptr = WordEmbedding_->GetSumGradient2EO(output_nodes[i]);
                    assert(ptr != nullptr);
                    AddRow(ptr, kSumGradient2EOTableId, output_nodes[i],
                        option_->embeding_size);
                    blocks.push_back(ptr);
                }
            }

            //Return all the memory blocks
            memory_mamanger_->ReturnBlocks(blocks);
        }


        void Trainer::SaveEmbedding(const char *file_path, bool is_binary)
        {
            FILE* fid = nullptr;
            if (is_binary)
            {
                fopen_s(&fid, file_path, "wb");
                fprintf_s(fid, "%d %d\n", dictionary_->Size(),option_->embeding_size);
                for (int i = 0; i < dictionary_->Size(); ++i)
                {
                    fprintf_s(fid, "%s ",
                        dictionary_->GetWordInfo(i)->word.c_str());

                    multiverso::Row<real>& embedding = GetRow<real>(
                        kInputEmbeddingTableId, i);

                    for (int j = 0; j < option_->embeding_size; ++j)
                    {
                        real tmp = embedding.At(j);
                        fwrite(&tmp, sizeof(real), 1, fid);
                    }

                    fprintf_s(fid, "\n");
                }

                fclose(fid);
            }
            else
            {
                fopen_s(&fid, file_path, "wt");
                fprintf_s(fid, "%d %d\n", dictionary_->Size(), option_->embeding_size);
                for (int i = 0; i < dictionary_->Size(); ++i)
                {
                    fprintf_s(fid, "%s ", dictionary_->GetWordInfo(i)->word.c_str());
                    multiverso::Row<real>& embedding = GetRow<real>(kInputEmbeddingTableId, i);

                    for (int j = 0; j < option_->embeding_size; ++j)
                        fprintf_s(fid, "%lf ", embedding.At(j));

                    fprintf_s(fid, "\n");
                }

                fclose(fid);
            }
        }
    }
}