#include <cstring>
#include <cmath>

#include <thread>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <new>

#include "multiverso/util/log.h"
#include "multiverso/multiverso.h"
#include "distributed_wordembedding.h"
#include "memory_manager.h"

#include "dictionary.h"
#include "huffman_encoder.h"
#include "util.h"
#include "reader.h"

using namespace multiverso;
using namespace wordembedding;

int main(int argc, char *argv[])
{
	try
	{
		Distributed_wordembedding *ptr = new (std::nothrow)Distributed_wordembedding();
		assert(ptr != nullptr);
		ptr->Run(argc, argv);
	}
	catch (std::bad_alloc &memExp)
	{
		multiverso::Log::Info("Something wrong with new() %s\n", memExp.what());
	}
	catch (...)
	{
		multiverso::Log::Info("Something wrong with other reason!\n");
	}
	return 0;
}
