#include "data_block.h"

namespace multiverso
{
	namespace wordembedding
	{
		DataBlock::~DataBlock()
		{
			ClearSentences();
		}

		size_t DataBlock::Size()
		{
			return sentences_.size();
		}

		//Add a new sentence to the DataBlock
		void DataBlock::AddSentence(int *head, int sentence_length,
			int64 word_count, uint64 next_random)
		{
			Sentence sentence(head, sentence_length, word_count, next_random);
			sentences_.push_back(sentence);
		}

		//Get the information of the index-th sentence
		void DataBlock::GetSentence(int index, int* &head,
			int &sentence_length, int64 &word_count, uint64 &next_random)
		{
			if (index >= 0 && index < sentences_.size())
			{
				sentences_[index].Get(head, sentence_length,
					word_count, next_random);
			}
			else
			{
				head = nullptr;
				sentence_length = 0;
				word_count = 0;
				next_random = 0;
			}
		}
	    //Free the memory of sentences
		void DataBlock::ClearSentences()
		{
			for (int i = 0; i < sentences_.size(); ++i)
				delete [] sentences_[i].head;
			sentences_.clear();
		}
	}
}