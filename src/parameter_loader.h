#pragma once

/*!
* file parameter_loader.h
* \brief Class Parameterloader parses the datablock and requests the params from multiverso server
*/

#include "multiverso.h"
#include "data_block.h"
#include "constant.h"
#include "util.h"
#include "huffman_encoder.h"
#include "wordembedding.h"
#include "log.h"

namespace multiverso
{
	namespace wordembedding
	{
		class WordEmbedding;
		extern std::string g_log_suffix;
		
		class ParameterLoader : public multiverso::ParameterLoaderBase
		{
		public:
			ParameterLoader(){}
			ParameterLoader(Option *option, WordEmbedding *WordEmbedding);
			/*!
			* \brief Parse the datablock to get the parameter needed
			* \param data_block which is pushed in
			*/
			void ParseAndRequest(multiverso::DataBlockBase* data_block) override;

		private:
			Option *option_;
			WordEmbedding *WordEmbedding_;
			int parse_and_request_count_;
			clock_t start_;
			FILE* log_file_;
			/*!
			* \brief Request the parameters from multiverso server to local buffer
			* \param data_block which is pushed in
			* \param input_nodes stores the input words'index
			* \param output_nodes stores the output words'index
            */
			void RequestParameter(DataBlock *data_block,
				std::vector<int>& input_nodes,
				std::vector<int>& output_nodes);
			//No copying allowed
			ParameterLoader(const ParameterLoader&);
			void operator=(const ParameterLoader&);
		};
	}
}

