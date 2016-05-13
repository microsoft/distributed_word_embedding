#ifndef DISTRIBUTED_WORD_EMBEDDING_CONSTANT_H_
#define DISTRIBUTED_WORD_EMBEDDING_CONSTANT_H_
#pragma once

/*!
* \file constant.h
* \brief The index of parameter tables and some constant.
*/

#include "multiverso/multiverso.h"
#include "multiverso/util/log.h"
#include <cstdint>

namespace multiverso
{
	namespace wordembedding
	{

		typedef int64_t int64;
		typedef uint64_t uint64;
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
#endif
