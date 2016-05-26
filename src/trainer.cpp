#include "trainer.h"
namespace multiverso
{
	namespace wordembedding
	{
		Trainer::Trainer(int trainer_id, Option *option,
			Dictionary* dictionary, WordEmbedding* WordEmbedding)
		{
			trainer_id_ = trainer_id;
			option_ = option;
			word_count = 0;
			WordEmbedding_ = WordEmbedding;
			dictionary_ = dictionary;
			hidden_act_ = (real *)calloc(option_->embeding_size, sizeof(real));
			hidden_err_ = (real *)calloc(option_->embeding_size, sizeof(real));
			process_count_ = -1;
			process_id_ = -1;

			assert(hidden_act_ != nullptr);
			assert(hidden_err_ != nullptr);
			start_ = 0;
			train_count_ = 0;
			/*
			if (trainer_id_ == 0)
			{
				//The log which recordes the begin and end time of TrainIteration()
				char log_name[100];
				sprintf(log_name, "trainer%s.txt", g_log_suffix.c_str());
				log_file_ = fopen(log_name, "w");
			}
			*/
		}

		void Trainer::TrainIteration(DataBlock *data_block)
		{
			if (process_id_ == -1)
				process_id_ = multiverso::MV_Rank();

			if (data_block == nullptr){
				return;
			}

			clock_t start = clock();

			multiverso::Log::Info("Rank %d Train %d TrainNN Begin TrainIteration%d ...\n",
				process_id_, trainer_id_, train_count_);

			WordEmbedding_->Train(data_block, trainer_id_, option_->thread_cnt,
				word_count, hidden_act_, hidden_err_);

			multiverso::Log::Info("Rank %d Trainer %d training time:%lfs\n",process_id_,trainer_id_,
				(clock() - start) / (double)CLOCKS_PER_SEC);
			train_count_++;
		}
	}
}
