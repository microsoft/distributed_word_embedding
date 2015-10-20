#include "distributed_wordembedding.h"
namespace multiverso
{
    namespace wordembedding
	{
        void Distributed_wordembedding::Train(int argc, char *argv[])
		{
			//The barrier for trainers
            multiverso::Barrier* barrier =
            new multiverso::Barrier(option_->thread_cnt);

            MemoryManager* memory_mamanger =
                new MemoryManager(option_->embeding_size);
            WordEmbedding* WordEmbeddings[2] =
            {   new WordEmbedding(option_, huffman_encoder_,
                    sampler_, dictionary_->Size()),
                new WordEmbedding(option_, huffman_encoder_,
                    sampler_, dictionary_->Size()) };

            //Step 1, Create Multiverso ParameterLoader and Trainers, 
            //Start Multiverso environment
            WordEmbeddings[1]->MallocMemory();

            //Prepare option_->thread_cnt trainers for multiverso
            std::vector<multiverso::TrainerBase*>trainers;
            for (int i = 0; i < option_->thread_cnt; ++i)
            {
                trainers.push_back(new (std::nothrow)Trainer(i, option_,
                    barrier, dictionary_, WordEmbeddings[1], memory_mamanger));
                assert(trainers[i] != nullptr);
            }

            //Start a thread to collect word_count from every trainers,
            //and update the WordEmbeddings[1]->word_count_actual
            StartCollectWordcountThread(trainers, WordEmbeddings[1]);

            //Prepare ParameterLoader
            ParameterLoader *parameter_loader =new (std::nothrow)ParameterLoader(
                option_, WordEmbeddings[0]);
            assert(parameter_loader != nullptr);

            //Step 2, prepare the Config for multiverso
            multiverso::Config config;
            config.max_delay = option_->max_delay;
            config.num_servers = option_->num_servers;
            config.num_aggregator = option_->num_aggregator;
            config.is_pipeline = option_->is_pipeline;
            config.lock_option =
                static_cast<multiverso::LockOption>(option_->lock_option);
            config.num_lock = option_->num_lock;
            //Config.server_endpoint_file = std::string(option_->endpoints_file);

            //Step3, Init the environment of multiverso
            multiverso::Multiverso::Init(trainers, parameter_loader,
                config, &argc, &argv);

            char log_name[100];
            sprintf(log_name, "log%s.txt", g_log_suffix.c_str());
            multiverso::Log::ResetLogFile(log_name);
            //Mark the node machine number
            process_id_ = multiverso::Multiverso::ProcessRank();
            //Step 4, prepare the sever/aggregator/cache Table for parametertable(3 or 5) 
            //and initialize the severtable for inputvector
            PrepareMultiversoParameterTables(option_, dictionary_);
			
            //Step 5, start the Train of NN
            TrainNeuralNetwork();

            //Step6, stop the thread which are collecting word_count,
            //and release the resource
            StopCollectWordcountThread();
            delete barrier;
            delete memory_mamanger;
            delete WordEmbeddings[0];
            delete WordEmbeddings[1];
            for (auto trainer : trainers)
            {
                delete trainer;
            }
            delete parameter_loader;
            multiverso::Multiverso::Close();
        }

        //The thread to collect word_count from trainers_
        void Distributed_wordembedding::StartThread()
        {
            int64 total_word_count = 0, sum = 0;
            while (is_running_)
            {
                sum = 0;
                for (int i = 0; i < trainers_.size(); ++i)
                        sum += trainers_[i]->word_count;

                if (sum < 10000 + total_word_count)
                {
                    std::chrono::milliseconds dura(20);
                    std::this_thread::sleep_for(dura);
                }
                else
                {
                    WordEmbedding_->word_count_actual += sum - total_word_count;
                    WordEmbedding_->UpdateLearningRate();
                    total_word_count = sum;

                    multiverso::Log::Info("Rank %d Alpha: %lf Progress: %.2lf%% WordCountActual: %lld Words/thread/second %lfk\n",
                        multiverso::Multiverso::ProcessRank(), WordEmbedding_->learning_rate,
                        WordEmbedding_->word_count_actual/ ((double)option_->total_words * option_->epoch + 1) * 100,
						WordEmbedding_->word_count_actual,
                        total_word_count / ((double)option_->thread_cnt * (clock()-start_)/ CLOCKS_PER_SEC * 1000.0));
                }
            }

            //Add the left word_count to the WordEmbedding
            WordEmbedding_->word_count_actual += sum - total_word_count;
            WordEmbedding_->UpdateLearningRate();
        }

        //Start a thread to collect the word count from trainers
        //The thread can be stopped by StopCollectWordcountThread()
        void Distributed_wordembedding::StartCollectWordcountThread(
            std::vector<multiverso::TrainerBase*> &trainer_bases, WordEmbedding *WordEmbedding)
        {
            is_running_ = true;
            WordEmbedding_ = WordEmbedding;
            for (int i = 0; i < trainer_bases.size(); ++i)
                trainers_.push_back(reinterpret_cast<Trainer*>(trainer_bases[i]));

            //Start a thread to collect the actual_word_count
            collect_wordcount_thread_ = std::thread(
                &Distributed_wordembedding::StartThread, this);
        }

        //Stop the thread which is collecting the word_count_actual from trainers
        void Distributed_wordembedding::StopCollectWordcountThread()
        {
            is_running_ = false;
            collect_wordcount_thread_.join();
        }

        //Create the three kinds of tables
        void Distributed_wordembedding::CreateMultiversoParameterTable(
            multiverso::integer_t table_id, multiverso::integer_t rows,
            multiverso::integer_t cols, multiverso::Type type,
            multiverso::Format default_format)
        {
            multiverso::Multiverso::AddServerTable(table_id, rows,
                cols, type, default_format);
            multiverso::Multiverso::AddCacheTable(table_id, rows,
                cols, type, default_format, 0);
            multiverso::Multiverso::AddAggregatorTable(table_id, rows,
                cols, type, default_format, 0);
        }

        void Distributed_wordembedding::PrepareMultiversoParameterTables(
            Option *opt, Dictionary *dictionary)
        {
            multiverso::Multiverso::BeginConfig();
            int proc_count = multiverso::Multiverso::TotalProcessCount();

            //Create tables, the order of creating tables should arise from 0 continuously
            //The elements of talbes will be initialized with 0
            CreateMultiversoParameterTable(kInputEmbeddingTableId,
                dictionary->Size(), opt->embeding_size,
                multiverso::Type::Float, multiverso::Format::Dense);

            CreateMultiversoParameterTable(kEmbeddingOutputTableId,
                dictionary->Size(), opt->embeding_size,
                multiverso::Type::Float, multiverso::Format::Dense);

            CreateMultiversoParameterTable(kWordCountActualTableId, 1, 1,
                multiverso::Type::LongLong, multiverso::Format::Dense);

            if (opt->use_adagrad)
            {
                CreateMultiversoParameterTable(kSumGradient2IETableId,
                    dictionary->Size(), opt->embeding_size,
                    multiverso::Type::Float, multiverso::Format::Dense);
                CreateMultiversoParameterTable(kSumGradient2EOTableId,
                    dictionary->Size(), opt->embeding_size,
                    multiverso::Type::Float, multiverso::Format::Dense);
            }

            //Initialize server tables
            //Every process will execute the code below, so the initialized
            //value should be divided by the number of processes
            for (int row = 0; row < dictionary->Size(); ++row)
            {
                for (int col = 0; col < opt->embeding_size; ++col)
                {
                    multiverso::Multiverso::AddToServer<real>(
                            kInputEmbeddingTableId, row, col,
                            static_cast<real>((static_cast<real>(rand())
                            / RAND_MAX - 0.5) / opt->embeding_size / proc_count));
                }
            }

            multiverso::Multiverso::EndConfig();
        }

        //Get the size of filename, it should deal with large files
        int64 Distributed_wordembedding::GetFileSize(const char *filename)
        {
#ifdef _MSC_VER
            struct _stat64 info;
            _stat64(filename, &info);
            return (int64)info.st_size;
#else
            struct  stat info;
            stat(filename, &info);
            return(int64)info.st_size;
#endif  
        }

        //Remove the datablock which is delt by parameterloader and trainer
        void Distributed_wordembedding::RemoveDoneDataBlock(
            std::queue<DataBlock*> &datablock_queue)
        {
            while (datablock_queue.empty() == false 
                && datablock_queue.front()->IsDone())
            {
                DataBlock *p_data_block = datablock_queue.front();
                datablock_queue.pop();
                delete p_data_block;
            }
        }

        void Distributed_wordembedding::PushDataBlock(
            std::queue<DataBlock*> &datablock_queue, DataBlock* data_block)
        {
			
            multiverso::Multiverso::PushDataBlock(data_block);
			
            datablock_queue.push(data_block);
            //limit the max size of total datablocks to avoid out of memory
            while (static_cast<int64>(datablock_queue.size()) * option_->data_block_size
                > option_->max_preload_data_size)
            {
                std::chrono::milliseconds dura(200);
                std::this_thread::sleep_for(dura);
                //Remove the datablock which is delt by parameterloader and trainer
                RemoveDoneDataBlock(datablock_queue);
            }
        }

        void Distributed_wordembedding::TrainNeuralNetwork()
        {
            std::queue<DataBlock*>datablock_queue;
            int data_block_count = 0;
            int64 file_size = GetFileSize(option_->train_file);
            multiverso::Log::Info("train-file-size:%lld, data_block_size:%lld\n",
                file_size, option_->data_block_size);
            start_ = clock();
            multiverso::Multiverso::BeginTrain();
            for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch)
            {
                reader_->ResetStart();
                multiverso::Multiverso::BeginClock();
                for (int64 cur = 0; cur < file_size; cur += option_->data_block_size)   
                {
                    ++data_block_count;
                    DataBlock *data_block = new (std::nothrow)DataBlock();
                    assert(data_block != nullptr);
                    //Load the sentences from train file, and store them in data_block
                    clock_t start = clock();
                    LoadData(data_block, reader_, option_->data_block_size);
                    multiverso::Log::Info("LoadOneDataBlockTime:%lfs\n",
                        (clock() - start) / (double)CLOCKS_PER_SEC);
                    PushDataBlock(datablock_queue, data_block);

                }
                multiverso::Multiverso::EndClock();
            }

            multiverso::Log::Info("Rank %d Pushed %d datablocks\n",
                process_id_, data_block_count);

            multiverso::Multiverso::EndTrain();

            //After EndTrain, all the datablock are done,
            //we remove all the datablocks
            RemoveDoneDataBlock(datablock_queue);
        }

        void Distributed_wordembedding::LoadData(DataBlock *data_block,
            Reader *reader, int64 size)
        {
            //Be sure to clear all the sentences
            //which were stored in data_block
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
			
            multiverso::Log::Info("Loading vocabulary ...\n");
            option_->total_words = LoadVocab(option_, dictionary_,
                huffman_encoder_);
            multiverso::Log::Info("Loaded vocabulary\n");

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
		
            delete reader_;
            delete sampler_;
            delete huffman_encoder_;
            delete dictionary_;
            delete option_;
        }

        //Readword from train_file to word array by the word index 
        bool Distributed_wordembedding::ReadWord(char *word, FILE *fin)
        {
            int idx = 0;
            char ch;
            while (!feof(fin))
            {
                ch = fgetc(fin);
                if (ch == 13) continue;
                if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
                {
                    if (idx > 0)
                    {
                        if (ch == '\n')
                            ungetc(ch, fin);
                        break;
                    }

                    if (ch == '\n')
                    {
                        strcpy(word, (char *)"</s>");
                        return true;
                    }
                    else
                    {
                        continue;
                    }
                }

                word[idx++] = ch;
                if (idx >= kMaxString - 1)
                    idx--;   
            }

            word[idx] = 0;
            return idx > 0;
        }

		 
        //Read the vocabulary file; create the dictionary
        //and huffman_encoder according opt
        int64 Distributed_wordembedding::LoadVocab(Option *opt,
            Dictionary *dictionary, HuffmanEncoder *huffman_encoder)
        {
            int64 total_words = 0;
            char word[kMaxString];
            FILE* fid = nullptr;
            multiverso::Log::Info("vocab_file %s\n", opt->read_vocab_file);
            if (opt->read_vocab_file != nullptr && strlen(opt->read_vocab_file) > 0)
            {
                multiverso::Log::Info("Begin to load vocabulary file [%s] ...\n",
                    opt->read_vocab_file);
                fid = fopen(opt->read_vocab_file, "r");
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
            if (opt->hs)
                huffman_encoder->BuildFromTermFrequency(dictionary);
            if (fid != nullptr)
                fclose(fid);

            return total_words;
        }
    }
}
