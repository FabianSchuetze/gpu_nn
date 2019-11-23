#pragma once
#ifndef cuda_workspace_manager_h
#define cuda_workspace_manager_h
#include <cudnn.h>
#include <iostream>
class WorkspaceManager {
   public:
    WorkspaceManager() : d_workspace(NULL), _bytes(0){};
    WorkspaceManager(size_t bytes) { cudaMalloc(&d_workspace, bytes); };
    ~WorkspaceManager() { cudaFree(d_workspace); };
    void* gpu_pointer() { return d_workspace; };
    size_t size() { return _bytes; };
    void resize(size_t bytes) {
        cudaFree(d_workspace);
        cudaMalloc(&d_workspace, bytes);
        _bytes = bytes;
    }
    void print() {
        std::cerr << "Workspace size: " << (_bytes / 1048576.0) << "MB"
                  << std::endl;
    }

   private:
    void* d_workspace;
    size_t _bytes;
};
#endif
