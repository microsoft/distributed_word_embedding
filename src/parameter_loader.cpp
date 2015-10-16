#include "parameter_loader.h"
namespace multiverso
{
    namespace wordembedding
    {
        ParameterLoader::ParameterLoader(Option *option,
            WordEmbedding *WordEmbedding)
        {
            option_ = option;
            WordEmbedding_ = WordEmbedding;

            parse_and_request_count_ = 0;

            //the log which will store the begin and end time of ParseAndRequest
            char log_name[100];
            sprintf_s(log_name, "parameter_loader%s.txt", g_log_suffix.c_str());
            fopen_s(&log_file_, log_name, "w");
        }
	
        void ParameterLoader::ParseAndRequest(
            multiverso::DataBlockBase *data_block)
        {
            if (parse_and_request_count_ == 0)
            {
                start_ = clock();
            }

            fprintf_s(log_file_, "%lf\n", (clock()) / (double)CLOCKS_PER_SEC);
            multiverso::Log::Info("Rank %d ParameterLoader begin %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            ++parse_and_request_count_;


            DataBlock *data = reinterpret_cast<DataBlock*>(data_block);
            //Step 1, compute the parameters which will be used when the trainers begin 
            std::vector<int> input_nodes;
            std::vector<int> output_nodes;
            //input_nodes,output_nodes
            multiverso::Log::Debug("Rank %d ParameterLoader parse begin %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            WordEmbedding_->PrepareParameter(data, input_nodes, output_nodes);
            multiverso::Log::Debug("Rank %d ParameterLoader parse end %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            //Step 2, Request the parameter
            multiverso::Log::Debug("Rank %d ParameterLoader request begin %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            RequestParameter(data, input_nodes, output_nodes);
            multiverso::Log::Debug("Rank %d ParameterLoader request end %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            //Step 3, store the needed parameters in data_block
            //it will be used to copy parameter from multiverso in trainer
            data->input_nodes = std::move(input_nodes);
            data->output_nodes = std::move(output_nodes);

            multiverso::Log::Info("Rank %d ParameterLoader finish %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_ - 1);
            fprintf_s(log_file_, "%lf\n", (clock()) / (double)CLOCKS_PER_SEC);
            fflush(log_file_);
        }

        void ParameterLoader::RequestParameter(DataBlock *data_block,
            std::vector<int>& input_nodes,
            std::vector<int>& output_nodes) 
        {
            //If the data_block is the last one, we need to dump 
            //the input-embedding weights
            if (data_block->Type() == DataBlockType::Test)
                RequestTable(kInputEmbeddingTableId);

            RequestRow(kWordCountActualTableId, 0);
            for (int i = 0; i < input_nodes.size(); ++i)
                RequestRow(kInputEmbeddingTableId, input_nodes[i]);
            for (int i = 0; i < output_nodes.size(); ++i)
                RequestRow(kEmbeddingOutputTableId, output_nodes[i]);
            if (option_->use_adagrad)
            {
                for (int i = 0; i < input_nodes.size(); ++i)
                    RequestRow(kSumGradient2IETableId, input_nodes[i]);
                for (int i = 0; i < output_nodes.size(); ++i)
                    RequestRow(kSumGradient2EOTableId, output_nodes[i]);
            }
        }   
    }
}