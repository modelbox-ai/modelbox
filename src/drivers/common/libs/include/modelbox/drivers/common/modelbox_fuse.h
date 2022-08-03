/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODELBOX_FUSE_H_
#define MODELBOX_FUSE_H_

#define _FILE_OFFSET_BITS 64
#define FUSE_USE_VERSION 31

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <fuse.h>
#include <modelbox/base/status.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace modelbox {

/**
 * @brief modelbox fuse file operations
 */
class ModelBoxFuseFile {
 public:
  /**
   * @brief Open file.
   * @param path file path.
   * @return open result.
   */
  virtual int Open(const std::string &path) = 0;

  /**
   * @brief Release file.
   * @return release result.
   */
  virtual int Release() = 0;

  /**
   * @brief read data from file.
   * @param buff read buffer.
   * @param size buffer size.
   * @param off current read offset.
   * @return read result.
   */
  virtual int Read(char *buff, size_t size, off_t off) = 0;

  /**
   * @brief write data to file.
   * @param buff data.
   * @param size data size.
   * @param off current write offset.
   * @return write result.
   */
  virtual int Write(const char *buff, size_t size, off_t off) = 0;

  /**
   * @brief sync file
   * @param isdatasync sync data?
   * @return sync result
   */
  virtual int FSync(int isdatasync) = 0;

  /**
   * @brief flush file
   * @return flush result
   */
  virtual int Flush() = 0;
};

/**
 * @brief modelbox fuse file type
 */
enum MODELBOX_FUSE_INODE_TYPE : unsigned int {

  /** @brief type none */
  MODELBOX_FUSE_INODE_TYPE_NONE = 0,
  /** @brief inode type file */
  MODELBOX_FUSE_INODE_TYPE_FILE = 1,
  /** @brief inode type directory */
  MODELBOX_FUSE_INODE_TYPE_DIR = 2,
};

class ModelBoxDEntry;

/**
 * @brief modelbox fuse file inode
 */
class ModelBoxInode {
 public:
  /**
   * @brief constructor
   */
  ModelBoxInode();

  /**
   * @brief destructor
   */
  virtual ~ModelBoxInode();

  /**
   * @brief fillup file stat
   * @param stat output parameter, file stat
   * @return fillup result, whether sucess or not.
   */
  virtual int FillStat(struct stat *stat);

  /**
   * @brief Get inode type
   * @return inode type
   */
  enum MODELBOX_FUSE_INODE_TYPE GetInodeType() { return inode_type_; }

  /**
   * @brief get DEntry
   * @return dentry
   */
  std::shared_ptr<ModelBoxDEntry> GetDEntry();

 protected:
  /**
   * @brief set inode type
   * @param type inode type
   */
  void SetInodeType(enum MODELBOX_FUSE_INODE_TYPE type) { inode_type_ = type; }

 private:
  friend ModelBoxDEntry;
  void SetDEntry(const std::shared_ptr<ModelBoxDEntry> &dentry);

  std::weak_ptr<ModelBoxDEntry> dentry_;
  enum MODELBOX_FUSE_INODE_TYPE inode_type_;
};

/**
 * @brief directory inode
 */
class ModelBoxDirInode : public ModelBoxInode {
 public:
  /**
   * @brief constructor
   */
  ModelBoxDirInode();

  /**
   * @brief destructor
   */
  ~ModelBoxDirInode() override;

  /**
   * @brief Store stat structure data
   * @brief stat stat data
   */
  void SetStat(struct stat *stat);

  /**
   * @brief fillup file stat
   * @param stat output parameter, file stat
   * @return fillup result, whether sucess or not.
   */
  int FillStat(struct stat *stat) override;

 private:
  struct stat stat_;
};

/**
 * @brief file inode
 */
class ModelBoxFileInode : public ModelBoxInode {
 public:
  /**
   * @brief constructor
   */
  ModelBoxFileInode();

  /**
   * @brief destructor
   */
  ~ModelBoxFileInode() override;

  /**
   * @brief  Create modelbox fuse file object
   * @return fuse file object
   */
  virtual std::shared_ptr<ModelBoxFuseFile> CreateFile() = 0;

  /**
   * @brief fillup file stat
   * @param stat output parameter, file stat
   * @return fillup result, whether sucess or not.
   */
  int FillStat(struct stat *stat) override = 0;

  /**
   * @brief Get current inode file path
   * @return inode path.
   */
  virtual std::string GetPath() = 0;
};

class ModelBoxDEntry : public std::enable_shared_from_this<ModelBoxDEntry> {
 public:
  /**
   * @brief constructor
   */
  ModelBoxDEntry();

  /**
   * @brief destructor
   */
  virtual ~ModelBoxDEntry();

  /**
   * @brief Find the DEntry structure according to the path
   * @param path fuse file path
   * @return DEntry object
   */
  std::shared_ptr<ModelBoxDEntry> LookUp(const std::string &path);

  /**
   * @brief Set parent DEntry
   * @param dentry dentry object
   * @return result
   */
  void SetParent(const std::shared_ptr<ModelBoxDEntry> &dentry);

  /**
   * @brief Add child DEntry
   * @param dentry dentry object
   * @return result
   */
  int AddChild(const std::shared_ptr<ModelBoxDEntry> &dentry);

  /**
   * @brief remove child DEntry by name
   * @param dentry dentry object
   * @return result
   */
  int RmvChild(const std::string &name);

  /**
   * @brief Get child entry number
   * @return number of child inodes
   */
  int ChildNum();

  /**
   * @brief Get child directory number
   * @return number of child inodes
   */
  int ChildDirNum();

  /**
   * @brief Get parent entry object
   * @return parent object, returns null if it does not exist
   */
  std::shared_ptr<ModelBoxDEntry> Parent();

  /**
   * @brief Get all children entry objects
   * @return vector of children objects
   */
  std::vector<std::shared_ptr<ModelBoxDEntry>> Children();

  /**
   * @brief Set current dentry name
   * @param name entry name
   */
  void SetName(const std::string &name);

  /**
   * @brief Get current dentry name
   * @return Entry name
   */
  const std::string &GetName();

  /**
   * @brief set inode to dentry
   * @param inode inode object
   */
  void SetInode(const std::shared_ptr<ModelBoxInode> &inode);

  /**
   * @brief Get inode from dentry
   * @return inode object
   */
  std::shared_ptr<ModelBoxInode> GetInode();

 private:
  std::shared_ptr<ModelBoxDEntry> LookUp(std::list<std::string> &names);
  std::string name_;
  std::shared_ptr<ModelBoxInode> inode_;
  std::weak_ptr<ModelBoxDEntry> parent_;
  std::map<std::string, std::shared_ptr<ModelBoxDEntry>> children_;
  int dir_num_{0};
  std::mutex children_lock_;
};

class ModelBoxFuseOperation;

/**
 * @brief modelbox fuse
 */
class ModelBoxFuse {
 public:
  /**
   * @brief destructor
   */
  virtual ~ModelBoxFuse();

  /**
   * @brief Add fuse file to modelbox fuse filesystem
   * @param fuse_file inode of fuse file
   * @return whether add success.
   */
  Status AddFuseFile(const std::shared_ptr<ModelBoxFileInode> &fuse_file);

  /**
   * @brief Remove fuse file form modelbox fuse filesystem
   * @param path file path
   * @return whether remove success.
   */
  Status RmvFuseFile(const std::string &path);

  /**
   * @brief Get mount pointer path
   * @return path
   */
  std::string GetMountPoint();

  /**
   * @brief Delete directory
   * @param path path
   */
  int RmDir(const char *path);

  /**
   * @brief make directory
   * @param path path
   */
  int MkDir(const char *path, mode_t mode);

  /**
   * @brief unlink file
   * @param path path
   */
  int Unlink(const char *path);

  /**
   * @brief Run fuse file
   */
  Status Run();

  /**
   * @brief stop fuse
   */
  void Stop();

 private:
  friend ModelBoxFuseOperation;
  ModelBoxFuse();

  /* Low level fuse API */
  Status InitLowLevelFuse();
  void DestroyLowLevelFuse();
  void SetMountPoint(const std::string &path);

  /* Fuse fops callback function */
  void *FuseInit(struct fuse_conn_info *conn);
  void FuseDestroy(void *eh);
  int GetAttr(const char *path, struct stat *stbuf);
  int Access(const char *path, int mask);
  int StatFS(const char *path, struct statvfs *stbuf);

  int OpenDir(const char *path, struct fuse_file_info *fi);
  int ReleaseDir(const char *path, struct fuse_file_info *fi);
  int ReadDir(const char *path, void *buff, fuse_fill_dir_t filler,
              off_t offset, struct fuse_file_info *fi);
  int Create(const char *path, mode_t mode, struct fuse_file_info *fi);
  int Open(const char *path, struct fuse_file_info *fi);
  int Release(const char *path, struct fuse_file_info *fi);
  int Read(const char *path, char *buff, size_t size, off_t off,
           struct fuse_file_info *fi);
  int Write(const char *path, const char *buff, size_t size, off_t off,
            struct fuse_file_info *fi);
  int FSync(const char *path, int isdatasync, struct fuse_file_info *fi);
  int Flush(const char *path, struct fuse_file_info *fi);

  /* fuse loop */
  void FuseLoop();
  void StopFuseLoop();
  void FillDefaultStat(struct stat *stbuf);
  struct fuse_chan *fuse_chan_{nullptr};
  struct fuse *fuse_{nullptr};
  bool is_running_{false};
  std::string mount_point_;
  std::thread loop_thread_;
  std::shared_ptr<ModelBoxDEntry> root_entry_;
};

class ModelBoxFuseOperation {
 public:
  /**
   * @brief constructor
   */
  ModelBoxFuseOperation();
  /**
   * @brief destructor
   */
  virtual ~ModelBoxFuseOperation();

  /**
   * @brief Create modelbox fuse file system
   * @param mount_path modelbox fuse mount point
   * @return modelbox fuse object
   */
  static std::shared_ptr<ModelBoxFuse> CreateFuse(
      const std::string &mount_path);

  static void DestroyFuse(ModelBoxFuse *modelbox_fuse);

  static void *FuseInit(struct fuse_conn_info *conn);
  static void FuseDestroy(void *eh);
  static int GetAttr(const char *path, struct stat *stbuf);
  static int Access(const char *path, int mask);
  static int StatFS(const char *path, struct statvfs *stbuf);

  static int RmDir(const char *path);
  static int MkDir(const char *path, mode_t mode);
  static int OpenDir(const char *path, struct fuse_file_info *fi);
  static int ReleaseDir(const char *path, struct fuse_file_info *fi);
  static int ReadDir(const char *path, void *buff, fuse_fill_dir_t filler,
                     off_t offset, struct fuse_file_info *fi);
  static int Unlink(const char *path);

  static int Create(const char *path, mode_t mode, struct fuse_file_info *fi);
  static int Open(const char *path, struct fuse_file_info *fi);
  static int Release(const char *path, struct fuse_file_info *fi);
  static int Read(const char *path, char *buff, size_t size, off_t off,
                  struct fuse_file_info *fi);
  static int Write(const char *path, const char *buff, size_t size, off_t off,
                   struct fuse_file_info *fi);
  static int FSync(const char *path, int isdatasync, struct fuse_file_info *fi);
  static int Flush(const char *path, struct fuse_file_info *fi);

  /**
   * @brief Get the current Fuse context
   * @return fuse object
   */
  static ModelBoxFuse *CurrentModleBoxFuse();

 private:
  static void FuseLoop();
  static std::map<std::string, ModelBoxFuse *> modelbox_fuses_;
  static std::mutex modelbox_fuses_lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_FUSE_H_
