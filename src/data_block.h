#pragma once

/*!
* \file data_block.h
* \brief Class DataBlock is to store the necessary data for trainer and param_loader
*/

#include "util.h"
#include "multiverso.h"
#include "huffman_encoder.h"
#include "constant.h"

namespace multiverso
{
	namespace wordembedding
	{
		/*!
		* \brief The class DataBlock stores train for trainer and param_loader
		*/
		class DataBlock : public multiverso::DataBlockBase
		{
		public:
			std::vector <int> input_nodes, output_nodes;
			DataBlock(){}
			~DataBlock();

			/*!
			* \brief Get the number of sentences stored in DataBlock
			* \return the number of sentences
			*/
			size_t Size();
			/*!
			* \brief Add a new sentence to the DataBlock
			* \param sentence the starting address of the sentence
			* \param sentence_length the length of the sentence
			* \param word_count the number of words when getting the
			*        sentence from train-file
			* \param next_random the seed for getting random number
			*/
			void AddSentence(int *sentence, int sentence_length,
				int64 word_count, uint64 next_random);
			/*!
			* \brief Get the information of the index-th sentence
			* \param index the id of the sentence
			* \param sentence the starting address of the sentence
			* \param sentence_length the length of the sentence
			* \param word_count the number of words when getting the
			*        sentence from train-file
			* \param next_random the seed for getting random number
			*/
			void GetSentence(int index, int* &sentence,
				int &sentence_length, int64 &word_count,
				uint64 &next_random);

			/*!
			* \brief Release the memory which are using to store sentences
			*/
			void ClearSentences();

		private:
			/*! 
			* \brief The information of sentences
			* head the head address which store the sentence
			* length the number of words in the sentence
			* word_count the real word count of the sentence
			* next_random the random seed
			*/
			struct Sentence
			{  
				int* head;
				int length;
				int64 word_count;
				uint64 next_random;
				Sentence(int *head, int length, int64 word_count,
					uint64 next_random) :head(head), length(length),
					word_count(word_count), next_random(next_random){}

				void Get(int* &local_head, int &sentence_length,
					int64 &local_word_count, uint64 &local_next_random)
				{
					local_head = head;
					sentence_length = length;
					local_word_count = word_count;
					local_next_random = next_random;
				}
			};

			/*! \brief Store the information of sentences*/
			std::vector <Sentence> sentences_;

			// No copying allowed
			DataBlock(const DataBlock&);
			void operator=(const DataBlock&);
		};
	}
}