#include <thread>
#include <string>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <new>

#include "dictionary.h"
#include "huffman_encoder.h"
#include "util.h"
#include "reader.h"
#include "multiverso.h"
#include "barrier.h"
#include "Distributed_wordembedding.h"
#include "parameter_loader.h"
#include "trainer.h"
#include "word_embedding.h"
#include "memory_manager.h"

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
    catch(...)
    {
        multiverso::Log::Info("Something wrong with other reason!\n");
    }
    return 0;
}
