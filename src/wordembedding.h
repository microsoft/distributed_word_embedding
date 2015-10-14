#pragma once

/*!
* file WordEmbedding.h
* \brief Class WordEmbedding includes some functions and parameters about TrainNN 
*/

#include <vector>
#include <cstring>
#include "util.h"
#include "multiverso.h"
#include "huffman_encoder.h"
#include "complex.h"
#include "constant.h"

namespace multiverso
{
	namespace wordembedding
	{
		class WordEmbedding
		{
		public:
			real learning_rate;
			int64 word_count_actual;

			WordEmbedding(Option* option, HuffmanEncoder* huffmanEncoder,
				Sampler* sampler, int dictionary_size);
			~WordEmbedding();
			/*!
            * \brief Create memory for weight_IE_ weight_EO_ sum_gradient2_IE_ sum_gradient2_EO_
		    */
			void MallocMemory();
			/*!
			* \brief TrainNN 
			* \param data_block represents the trainNNing datablock 
			* \param index_start the thread's starting index in the sentence vector
			* \param interval the total_number of thread
			* \param word_count count the words which has been processed by trainNN
			* \param hidden_act  hidden layer value
			* \param hidden_err  hidden layer error
			*/
			void Train(DataBlock *data_block, int index_start,
				int interval, int64& word_count,
				real* hidden_act, real* hidden_err);
			/*!
			* \brief PrepareParameter for parameterloader threat
			* \param data_block datablock for parameterloader to parse
			* \param input_nodes  input_nodes represent the parameter which input_layer includes 
			* \param output_nodes output_nodes represent the parameter which output_layer inclueds
			*/
			void PrepareParameter(DataBlock *data_block,
				std::vector<int>& input_nodes, std::vector<int>& output_nodes);
			/*!
			* \brief Update the learning rate
			*/
            void UpdateLearningRate();
			/*!
			* \brief Set the input(output)-embeddding weight
			*/
			void SetWeightIE(int input_node_id, real* ptr);
			void SetWeightEO(int output_node_id, real* ptr);
			/*!
			* \brief Set the SumGradient-input(ouput) 
			*/
			void SetSumGradient2IE(int input_node_id, real* ptr);
			void SetSumGradient2EO(int output_node_id, real* ptr);
			/*!
			* \brief Return the parametertable value
			*/
			real* GetWeightIE(int input_node_id);
			real* GetWeightEO(int output_node_id);
			real* GetSumGradient2IE(int input_node_id);
			real* GetSumGradient2EO(int output_node_id);

		private:
			Option *option_;
			Dictionary *dictionary_;
			HuffmanEncoder *huffmanEncoder_;
			Sampler *sampler_;
			std::unordered_set<int> input_nodes_, output_nodes_;
			int dictionary_size_;
			real** weight_IE_;
			real** weight_EO_;
			real** sum_gradient2_IE_;
			real** sum_gradient2_EO_;

			typedef void(WordEmbedding::*FunctionType)(std::vector<int>& input_nodes,
				std::vector<std::pair<int, int> >& output_nodes,
				void *hidden_act, void *hidden_err);
			/*!
			* \brief Parse the needed parameter in a window
			*/
			void Parse(int *feat, int feat_cnt, int word_idx, uint64 &next_random,
				std::vector<int>& input_nodes,
				std::vector<std::pair<int, int> >& output_nodes);
			/*!
			* \brief Parse a sentence and deepen into two branchs
			* \one for TrainNN,the other one is for Parameter_parse&request
			*/
			void ParseSentence(int* sentence, int sentence_length,
				uint64 next_random,
				real* hidden_act, real* hidden_err,
				FunctionType function);
			/*!
			* \brief Get the hidden layer vector
			* \param input_nodes represent the input nodes
			* \param hidden_act store the hidden layer vector
			*/
			void FeedForward(std::vector<int>& input_nodes, real* hidden_act);
			/*!
			* \brief Calculate the hidden_err and update the output-embedding weight
			* \param label record the label of every output-embedding vector
			* \param word_idx the index of the output-embedding vector
			* \param classifier store the output-embedding vector
			* \param store the hidden layer vector
			* \param store the hidden-error which is used 
			* \to update the input-embedding vector
			*/
			void BPOutputLayer(int label, int word_idx, real* classifier,
				real* hidden_act, real* hidden_err);
			/*!
			* \brief Copy the input_nodes&output_nodes to WordEmbedding private set
			*/
			void DealPrepareParameter(std::vector<int>& input_nodes,
				std::vector<std::pair<int, int> >& output_nodes,
				void *hidden_act, void *hidden_err);
			/*!
			* \brief Train a window sample and update the 
			* \input-embedding&output-embedding vectors 
			* \param input_nodes represent the input nodes
			* \param output_nodes represent the ouput nodes 
			* \param hidden_act  store the hidden layer vector
			* \param hidden_err  store the hidden layer error
			*/
			void TrainSample(std::vector<int>& input_nodes,
				std::vector<std::pair<int, int> >& output_nodes,
				void *hidden_act, void *hidden_err);
			/*!
			* \brief Train the sentence actually
			*/
			void Train(int* sentence, int sentence_length,
				uint64 next_random, real* hidden_act, real* hidden_err);

			//No copying allowed
			WordEmbedding(const WordEmbedding&);
			void operator=(const WordEmbedding&);
		};
	}
}
