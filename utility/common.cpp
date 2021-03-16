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

#include "common.h"
#include <cstring>
#include <vector>
#include <cstdlib>
#include <dirent.h>

int min(int a, int b)
{
    return (a < b) ? a : b;
}

void* _mm_malloc(size_t sz, size_t align)
{
    void *ptr;
#ifdef __APPLE__
    return malloc(sz);
#else
    int alloc_result = posix_memalign(&ptr, align, sz);
    if (alloc_result != 0)
    {
        return NULL;
    }
    return ptr;
#endif
}

void _mm_free(void* ptr)
{
    if (NULL != ptr)
    {
        free(ptr);
        ptr = NULL;
    }
}

// 向上对齐
int align_ceil(int num, int align)
{
    return num + (align - (num % align)) % align;
}

int get_cache_info(size_t &l1_cache_size_per_core, size_t &l2_cache_size_per_core) {
#ifdef __APPLE__
    FILE* fp;
    int config;
    size_t size;
    std::vector<int> cache_config;
    std::vector<size_t> cache_size;
    if ((fp = popen("sysctl -n hw.cacheconfig", "r")) != nullptr)
        while (fscanf(fp, "%d", &config) != -1 && config != 0) {
            cache_config.push_back(config);
        }
    else 
        printf("get cacheconfig error!\n");
    pclose(fp);
    if ((fp = popen("sysctl -n hw.cachesize", "r")) != nullptr)
        while (fscanf(fp, "%u", &size) != -1 && size != 0)
            cache_size.push_back(size);
    else
        printf("get cachesize error!\n");
    pclose(fp);

    int cache_idx = cache_config.size() - 1;
    if (cache_idx >= 1) {
        l2_cache_size_per_core = cache_size[cache_idx] / static_cast<size_t>(cache_config[cache_idx]);
        l1_cache_size_per_core = cache_size[cache_idx - 1] / static_cast<size_t>(cache_config[cache_idx - 1]);
        printf("%u\n", l2_cache_size_per_core);
        printf("%u\n", l1_cache_size_per_core);
    } else if (cache_idx == 0) {
        l2_cache_size_per_core = 0;
        l1_cache_size_per_core = cache_size[cache_idx] / static_cast<size_t>(cache_config[cache_idx]);
    } else {
        l2_cache_size_per_core = 0;
        l1_cache_size_per_core = 0;
    }
    return 1;
#else
    size_t l1_cache_size = 0;
    size_t l2_cache_size = 0;
    const char *path = "/sys/devices/system/cpu/cpu0/cache";
    char child_path[256];
    char cache_info_file[256];
    DIR *directory_ptr;
    DIR *child_directory_ptr;
    struct dirent *entry;
    struct dirent *child_entry;
    struct CacheInfo cache_info;
    directory_ptr = opendir(path);
    if (directory_ptr == nullptr) {
        printf("The cache file dosen't exist.\n");
        return -1;
    }
    while ((entry = readdir(directory_ptr)) != nullptr) {   // traverse all cache level
        if (entry->d_type & DT_DIR) {
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                memset(child_path, 0, sizeof(child_path));
                sprintf(child_path,"%s/%s", path, entry->d_name);
                child_directory_ptr = opendir(child_path);
                if (child_directory_ptr == nullptr) {
                    printf("The cache file dosen't exist.\n");
                    return -1;
                } 
                char type[20];
                char shared_cpu_list[20];
                FILE *fp;
                cache_info.level = 0;
                cache_info.size  = 0;
                while ((child_entry = readdir(child_directory_ptr)) != nullptr) {   // traverse all param of a cache level
                    if (!(child_entry->d_type & DT_DIR)) {
                        if (strcmp(child_entry->d_name, "level") == 0) {    // level
                            memset(cache_info_file, 0, sizeof(cache_info_file));
                            sprintf(cache_info_file, "%s/%s", child_path, child_entry->d_name);
                            if ((fp = fopen(cache_info_file, "r")) != nullptr)
                                fscanf(fp, "%d", &(cache_info.level));
                            fclose(fp);
                        } else if (strcmp(child_entry->d_name, "size") == 0) {  // cache size
                            memset(cache_info_file, 0, sizeof(cache_info_file));
                            sprintf(cache_info_file, "%s/%s", child_path, child_entry->d_name);
                            if ((fp = fopen(cache_info_file, "r")) != nullptr)
                                fscanf(fp, "%d", &(cache_info.size));
                            fclose(fp);
                        } else if (strcmp(child_entry->d_name, "type") == 0) {  // cache type(data, instructions or unified)
                            memset(cache_info_file, 0, sizeof(cache_info_file));
                            memset(cache_info.type, 0, sizeof(cache_info.type));
                            sprintf(cache_info_file, "%s/%s", child_path, child_entry->d_name);
                            if ((fp = fopen(cache_info_file, "r")) != nullptr)
                                fscanf(fp, "%s", (cache_info.type));
                            fclose(fp);
                        } else if (strcmp(child_entry->d_name, "shared_cpu_list") == 0) {   // shared core
                            memset(cache_info_file, 0, sizeof(cache_info_file));
                            sprintf(cache_info_file, "%s/%s", child_path, child_entry->d_name);
                            if ((fp = fopen(cache_info_file, "r")) != nullptr) {
                                fscanf(fp, "%s", shared_cpu_list);
                                char* p = strtok(shared_cpu_list, "-");
                                int begin = 0, end = 0;
                                if (p != nullptr) {
                                    begin = atoi(p);
                                    p = strtok(nullptr, "-");
                                    if (p != nullptr) {
                                        end = atoi(p);
                                    }
                                }
                                cache_info.num_shared_core = abs(end - begin) + 1;
                            }
                            fclose(fp);
                        }
                    }
                }
                if (strcmp(type, "Instruction") != 0) {
                    if (cache_info.level == 1) {
                        if (cache_info.size == 0)
                            cache_info.size = 32;
                        l1_cache_size = cache_info.size * 1024;
                        l1_cache_size_per_core = l1_cache_size / (size_t)cache_info.num_shared_core;
                    } else if (cache_info.level == 2) {
                        if (cache_info.size == 0)
                            cache_info.size = 512;
                        l2_cache_size = cache_info.size * 1024;
                        l2_cache_size_per_core = l2_cache_size / (size_t)cache_info.num_shared_core;
                    }
                }
            }
        }
    }
    return 1;
#endif
}
