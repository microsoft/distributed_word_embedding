#pragma once

/*!
* \file constant.h
* \brief The index of parameter tables and some constant.
*/

#include "multiverso.h"
#include "log.h"


namespace multiverso
{
    namespace wordembedding
    {
        /*! \brief Table id is use*/
        const multiverso::integer_t kInputEmbeddingTableId = 0;
        const multiverso::integer_t kEmbeddingOutputTableId = 1;
        const multiverso::integer_t kWordCountActualTableId = 2;
        const multiverso::integer_t kSumGradient2IETableId = 3;
        const multiverso::integer_t kSumGradient2EOTableId = 4;

        typedef long long int64;
        typedef unsigned long long uint64;
        typedef float real;

        const int kTableSize = (int)1e8;
        const real kEps = static_cast<real>(1e-10);
        const int kMaxWordSize = 901;
        const int kMaxCodeLength = 100;
        const int kMaxString = 100;
        const int kMaxSentenceLength = 1000;
        const int kMaxEXP = 6;
    }
}
