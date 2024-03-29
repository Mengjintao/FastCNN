//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <pthread.h>

class StringTool
{
    public:
        static void SplitString(const std::string &input, const std::string &delim, std::vector<std::string> &parts);
};

struct CacheInfo {
    size_t size;
    int level;
    int num_shared_core;
    char type[32];
};

int min(int a, int b);
void* _mm_malloc(size_t sz, size_t align=64);
void _mm_free(void* ptr);
// 向上取整为align的倍数
int align_ceil(int num, int align);
int get_cache_info(size_t &l1_cache_size_per_core, size_t &l2_cache_size_per_core);