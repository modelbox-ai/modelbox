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

#include "modelbox/drivers/common/modelbox_fuse.h"

#include <libgen.h>
#include <modelbox/base/log.h>
#include <modelbox/base/os.h>
#include <modelbox/base/utils.h>
#include <unistd.h>

#include <ostream>
#include <sstream>
#include <thread>
#include <utility>

namespace modelbox {

std::list<std::string> SplitPath(const std::string &s, char delim) {
  std::list<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim)) {
    result.push_back(item);
  }

  return result;
}

std::map<std::string, ModelBoxFuse *> ModelBoxFuseOperation::modelbox_fuses_;
std::mutex ModelBoxFuseOperation::modelbox_fuses_lock_;

fuse_operations kModelboxFuseOperation = [] {
  fuse_operations ops{};
  ops.init = ModelBoxFuseOperation::FuseInit;
  ops.destroy = ModelBoxFuseOperation::FuseDestroy;
  ops.getattr = ModelBoxFuseOperation::GetAttr;
  ops.access = ModelBoxFuseOperation::Access;
  ops.statfs = ModelBoxFuseOperation::StatFS;

  ops.rmdir = ModelBoxFuseOperation::RmDir;
  ops.mkdir = ModelBoxFuseOperation::MkDir;
  ops.opendir = ModelBoxFuseOperation::OpenDir;
  ops.releasedir = ModelBoxFuseOperation::ReleaseDir;
  ops.readdir = ModelBoxFuseOperation::ReadDir;
  ops.unlink = ModelBoxFuseOperation::Unlink;

  ops.create = ModelBoxFuseOperation::Create;
  ops.open = ModelBoxFuseOperation::Open;
  ops.release = ModelBoxFuseOperation::Release;
  ops.read = ModelBoxFuseOperation::Read;
  ops.write = ModelBoxFuseOperation::Write;
  ops.fsync = ModelBoxFuseOperation::FSync;
  ops.flush = ModelBoxFuseOperation::Flush;
  return ops;
}();

void *ModelBoxFuseOperation::FuseInit(struct fuse_conn_info *conn) {
  conn->want |= FUSE_CAP_ATOMIC_O_TRUNC;
  conn->want |= FUSE_CAP_ASYNC_READ;
  CurrentModleBoxFuse()->FuseInit(conn);
  return CurrentModleBoxFuse();
}

void ModelBoxFuseOperation::FuseDestroy(void *eh) {
  CurrentModleBoxFuse()->FuseDestroy(eh);
}

int ModelBoxFuseOperation::GetAttr(const char *path, struct stat *stbuf) {
  return CurrentModleBoxFuse()->GetAttr(path, stbuf);
}

int ModelBoxFuseOperation::Access(const char *path, int mask) {
  return CurrentModleBoxFuse()->Access(path, mask);
}

int ModelBoxFuseOperation::StatFS(const char *path, struct statvfs *stbuf) {
  return CurrentModleBoxFuse()->StatFS(path, stbuf);
}

int ModelBoxFuseOperation::RmDir(const char *path) {
  return CurrentModleBoxFuse()->RmDir(path);
}

int ModelBoxFuseOperation::MkDir(const char *path, mode_t mode) {
  return CurrentModleBoxFuse()->MkDir(path, mode);
}

int ModelBoxFuseOperation::OpenDir(const char *path,
                                   struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->OpenDir(path, fi);
}

int ModelBoxFuseOperation::ReleaseDir(const char *path,
                                      struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->ReleaseDir(path, fi);
}

int ModelBoxFuseOperation::ReadDir(const char *path, void *buff,
                                   fuse_fill_dir_t filler, off_t offset,
                                   struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->ReadDir(path, buff, filler, offset, fi);
}

int ModelBoxFuseOperation::Unlink(const char *path) {
  return CurrentModleBoxFuse()->Unlink(path);
}

int ModelBoxFuseOperation::Open(const char *path, struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Open(path, fi);
}

int ModelBoxFuseOperation::Create(const char *path, mode_t mode,
                                  struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Create(path, mode, fi);
}

int ModelBoxFuseOperation::Release(const char *path,
                                   struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Release(path, fi);
}

int ModelBoxFuseOperation::Read(const char *path, char *buff, size_t size,
                                off_t off, struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Read(path, buff, size, off, fi);
}

int ModelBoxFuseOperation::Write(const char *path, const char *buff,
                                 size_t size, off_t off,
                                 struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Write(path, buff, size, off, fi);
}

int ModelBoxFuseOperation::FSync(const char *path, int isdatasync,
                                 struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->FSync(path, isdatasync, fi);
}
int ModelBoxFuseOperation::Flush(const char *path, struct fuse_file_info *fi) {
  return CurrentModleBoxFuse()->Flush(path, fi);
}

ModelBoxFuse *ModelBoxFuseOperation::CurrentModleBoxFuse() {
  return static_cast<ModelBoxFuse *>(fuse_get_context()->private_data);
}

std::shared_ptr<ModelBoxFuse> ModelBoxFuseOperation::CreateFuse(
    const std::string &mount_path) {
  std::shared_ptr<ModelBoxFuse> modelbox_fuse(new ModelBoxFuse);
  fuse_unmount(mount_path.c_str(), nullptr);
  modelbox_fuses_lock_.lock();
  modelbox_fuses_[mount_path] = modelbox_fuse.get();
  modelbox_fuses_lock_.unlock();

  modelbox_fuse->SetMountPoint(mount_path);
  return modelbox_fuse;
}

void ModelBoxFuseOperation::DestroyFuse(ModelBoxFuse *modelbox_fuse) {
  modelbox_fuses_lock_.lock();
  modelbox_fuses_.erase(modelbox_fuse->GetMountPoint());
  modelbox_fuses_lock_.unlock();
}

ModelBoxFuse::ModelBoxFuse() = default;
ModelBoxFuse::~ModelBoxFuse() {
  StopFuseLoop();
  DestroyLowLevelFuse();
}

Status ModelBoxFuse::InitLowLevelFuse() {
  const char *argv[] = {"modelbox", "-o", "nonempty"};
  const int argc = sizeof(argv) / sizeof(char *);
  struct fuse_args args = FUSE_ARGS_INIT(argc, (char **)argv);
  CreateDirectory(mount_point_);
  struct fuse_chan *chan = fuse_mount(mount_point_.c_str(), &args);
  if (chan == nullptr) {
    std::string err = "mount directory ";
    err += mount_point_ + " failed, ";
    err += StrError(errno);
    MBLOG_ERROR << err;
    if (errno == ENOENT) {
      return {STATUS_NOENT, err};
    }

    if (errno == EACCES) {
      return {STATUS_PERMIT, err};
    }

    return {STATUS_FAULT, err};
  }

  auto *fuse = fuse_new(chan, &args, &kModelboxFuseOperation,
                        sizeof(kModelboxFuseOperation), this);
  if (fuse == nullptr) {
    fuse_unmount(mount_point_.c_str(), chan);
    return {STATUS_FAULT, "new fuse failed."};
  }

  fuse_chan_ = chan;
  fuse_ = fuse;
  return STATUS_OK;
}

void ModelBoxFuse::DestroyLowLevelFuse() {
  if (fuse_) {
    fuse_destroy(fuse_);
    fuse_ = nullptr;
  }

  if (fuse_chan_) {
    fuse_unmount(mount_point_.c_str(), fuse_chan_);
    fuse_chan_ = nullptr;
  }

  rmdir(mount_point_.c_str());
}

void ModelBoxFuse::SetMountPoint(const std::string &path) {
  mount_point_ = path;
}

Status ModelBoxFuse::AddFuseFile(
    const std::shared_ptr<ModelBoxFileInode> &fuse_file) {
  auto path = fuse_file->GetPath();
  auto entry = root_entry_->LookUp(path);
  if (entry) {
    return STATUS_EXIST;
  }

  std::string parent = GetDirName(path);
  std::string dir = GetBaseName(path);

  auto parent_entry = root_entry_->LookUp(parent);
  if (parent_entry == nullptr) {
    return STATUS_NOENT;
  }

  entry = std::make_shared<ModelBoxDEntry>();
  entry->SetName(dir);
  entry->SetInode(fuse_file);
  parent_entry->AddChild(entry);
  return STATUS_OK;
}

Status ModelBoxFuse::RmvFuseFile(const std::string &path) {
  auto entry = root_entry_->LookUp(path);
  if (!entry) {
    return STATUS_NOENT;
  }

  std::string parent = GetDirName(path);
  std::string name = GetBaseName(path);

  auto parent_entry = root_entry_->LookUp(parent);
  if (parent_entry == nullptr) {
    return STATUS_NOENT;
  }

  if (parent_entry->RmvChild(name) != 0) {
    return STATUS_FAULT;
  }

  return STATUS_OK;
}

std::string ModelBoxFuse::GetMountPoint() { return mount_point_; }

void *ModelBoxFuse::FuseInit(struct fuse_conn_info *conn) { return nullptr; }
void ModelBoxFuse::FuseDestroy(void *eh) {}
int ModelBoxFuse::GetAttr(const char *path, struct stat *stbuf) {
  auto entry = root_entry_->LookUp(path);
  if (entry == nullptr) {
    return -ENOENT;
  }

  auto inode = entry->GetInode();
  FillDefaultStat(stbuf);
  switch (inode->GetInodeType()) {
    case MODELBOX_FUSE_INODE_TYPE_FILE:
      stbuf->st_mode |= S_IFREG;
      stbuf->st_mode |= 0440;
      break;
    case MODELBOX_FUSE_INODE_TYPE_DIR:
      stbuf->st_mode |= S_IFDIR;
      stbuf->st_mode |= 0750;
      break;
    default:
      break;
  }
  inode->FillStat(stbuf);
  stbuf->st_blocks = stbuf->st_size / 512;
  return 0;
}

int ModelBoxFuse::Access(const char *path, int mask) { return 0; }
int ModelBoxFuse::StatFS(const char *path, struct statvfs *stbuf) { return 0; }

int ModelBoxFuse::RmDir(const char *path) {
  MBLOG_DEBUG << "rm dir: " << path;

  auto entry = root_entry_->LookUp(path);
  if (!entry) {
    return -ENOENT;
  }

  std::string parent = GetDirName(path);
  std::string dir = GetBaseName(path);

  auto parent_entry = root_entry_->LookUp(parent);
  if (parent_entry == nullptr) {
    return -ENOENT;
  }

  parent_entry->RmvChild(dir);

  return 0;
}

int ModelBoxFuse::MkDir(const char *path, mode_t mode) {
  MBLOG_DEBUG << "make dir: " << path;

  auto entry = root_entry_->LookUp(path);
  if (entry) {
    return -EEXIST;
  }

  std::string parent = GetDirName(path);
  std::string dir = GetBaseName(path);

  auto parent_entry = root_entry_->LookUp(parent);
  if (parent_entry == nullptr) {
    return -ENOENT;
  }

  auto inode = std::make_shared<ModelBoxDirInode>();
  struct stat st;
  FillDefaultStat(&st);
  st.st_mode = mode;
  st.st_size = 4096;
  st.st_nlink = 2;
  inode->SetStat(&st);
  entry = std::make_shared<ModelBoxDEntry>();
  entry->SetName(dir);
  entry->SetInode(inode);
  parent_entry->AddChild(entry);

  return 0;
}

int ModelBoxFuse::OpenDir(const char *path, struct fuse_file_info *fi) {
  auto entry = root_entry_->LookUp(path);
  if (entry == nullptr) {
    return -ENOENT;
  }

  auto *holder = new std::shared_ptr<ModelBoxDEntry>;
  *holder = entry;
  fi->fh = (uint64_t)holder;
  return 0;
}

int ModelBoxFuse::ReleaseDir(const char *path, struct fuse_file_info *fi) {
  auto *entry = (std::shared_ptr<ModelBoxDEntry> *)(fi->fh);
  delete entry;
  return 0;
}

int ModelBoxFuse::ReadDir(const char *path, void *buff, fuse_fill_dir_t filler,
                          off_t offset, struct fuse_file_info *fi) {
  auto *entry = (std::shared_ptr<ModelBoxDEntry> *)(fi->fh);

  for (const auto &child : (*entry)->Children()) {
    struct stat st;
    auto inode = child->GetInode();
    if (inode == nullptr) {
      continue;
    }
    inode->FillStat(&st);
    int res = filler(buff, child->GetName().c_str(), &st, 0);
    if (res != 0) {
      MBLOG_WARN << "fill stat failed for " << child->GetName();
    }
  }

  const char *const dots[] = {".", ".."};

  for (const auto *str : dots) {
    struct stat st;
    FillDefaultStat(&st);
    int res = filler(buff, str, &st, 0);
    if (res != 0) {
      MBLOG_WARN << "fill stat failed for " << str;
    }
  }
  return 0;
}
int ModelBoxFuse::Unlink(const char *path) {
  MBLOG_DEBUG << "unlink file: " << path;

  auto ret = RmvFuseFile(path);

  if (ret == STATUS_NOENT) {
    return -ENOENT;
  }

  return 0;
}

int ModelBoxFuse::Create(const char *path, mode_t mode,
                         struct fuse_file_info *fi) {
  MBLOG_DEBUG << "create file: " << path;
  return -ENOSYS;
}

int ModelBoxFuse::Open(const char *path, struct fuse_file_info *fi) {
  MBLOG_DEBUG << "open file: " << path;

  auto entry = root_entry_->LookUp(path);
  if (entry == nullptr) {
    return -ENOENT;
  }

  auto inode = entry->GetInode();
  if (inode == nullptr) {
    return -EBADFD;
  }

  if (inode->GetInodeType() != MODELBOX_FUSE_INODE_TYPE_FILE) {
    return -EBADFD;
  }

  auto file_inode = std::dynamic_pointer_cast<ModelBoxFileInode>(inode);
  if (file_inode == nullptr) {
    return -EBADFD;
  }

  auto file_ops = file_inode->CreateFile();
  if (file_ops == nullptr) {
    return -ENOMEM;
  }

  int ret = file_ops->Open(path);
  if (ret != 0) {
    return ret;
  }

  auto *holder = new std::shared_ptr<ModelBoxFuseFile>;
  *holder = file_ops;
  fi->fh = (uint64_t)holder;
  return 0;
}

int ModelBoxFuse::Release(const char *path, struct fuse_file_info *fi) {
  auto *fuse_file = (std::shared_ptr<ModelBoxFuseFile> *)(fi->fh);
  int ret = (*fuse_file)->Release();
  delete fuse_file;
  MBLOG_DEBUG << "close file: " << path;
  return ret;
}

int ModelBoxFuse::Read(const char *path, char *buff, size_t size, off_t off,
                       struct fuse_file_info *fi) {
  auto *fuse_file = (std::shared_ptr<ModelBoxFuseFile> *)(fi->fh);
  return (*fuse_file)->Read(buff, size, off);
}

int ModelBoxFuse::Write(const char *path, const char *buff, size_t size,
                        off_t off, struct fuse_file_info *fi) {
  auto *fuse_file = (std::shared_ptr<ModelBoxFuseFile> *)(fi->fh);
  return (*fuse_file)->Write(buff, size, off);
}

int ModelBoxFuse::FSync(const char *path, int isdatasync,
                        struct fuse_file_info *fi) {
  auto *fuse_file = (std::shared_ptr<ModelBoxFuseFile> *)(fi->fh);
  return (*fuse_file)->FSync(isdatasync);
}

int ModelBoxFuse::Flush(const char *path, struct fuse_file_info *fi) {
  auto *fuse_file = (std::shared_ptr<ModelBoxFuseFile> *)(fi->fh);
  return (*fuse_file)->Flush();
}

Status ModelBoxFuse::Run() {
  auto ret = InitLowLevelFuse();
  if (!ret) {
    return ret;
  }

  MBLOG_DEBUG << "fuse " << mount_point_ << " start";
  is_running_ = true;
  root_entry_ = std::make_shared<ModelBoxDEntry>();
  root_entry_->SetName("/");
  auto inode = std::make_shared<ModelBoxDirInode>();

  struct stat stbuf;
  FillDefaultStat(&stbuf);
  stbuf.st_size = 4096;
  stbuf.st_mode |= S_IFDIR;
  stbuf.st_nlink = 2;
  inode->SetStat(&stbuf);
  root_entry_->SetInode(inode);

  loop_thread_ = std::thread(&ModelBoxFuse::FuseLoop, this);

  return STATUS_OK;
}

void ModelBoxFuse::FillDefaultStat(struct stat *stbuf) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  stbuf->st_mode = 0440;
  stbuf->st_mtim = ts;
  stbuf->st_atim = ts;
  stbuf->st_uid = getuid();
  stbuf->st_gid = getgid();
  stbuf->st_blksize = 4096;
  stbuf->st_size = 0;
  stbuf->st_nlink = 1;
  stbuf->st_blocks = stbuf->st_size / 512;
}

void ModelBoxFuse::Stop() { StopFuseLoop(); }

void ModelBoxFuse::FuseLoop() {
  os->Thread->SetName("FuseDaemon");
  fuse_loop_mt(fuse_);
}

void ModelBoxFuse::StopFuseLoop() {
  if (is_running_ == false) {
    return;
  }

  MBLOG_DEBUG << "fuse " << mount_point_ << " stop";

  is_running_ = false;
  ModelBoxFuseOperation::DestroyFuse(this);
  fuse_exit(fuse_);
  fuse_unmount(mount_point_.c_str(), fuse_chan_);
  fuse_chan_ = nullptr;

  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }

  DestroyLowLevelFuse();
}

ModelBoxInode::ModelBoxInode() = default;

ModelBoxInode::~ModelBoxInode() = default;

int ModelBoxInode::FillStat(struct stat *stat) { return 0; };

void ModelBoxInode::SetDEntry(const std::shared_ptr<ModelBoxDEntry> &dentry) {
  dentry_ = dentry;
}

std::shared_ptr<ModelBoxDEntry> ModelBoxInode::GetDEntry() {
  return dentry_.lock();
}

ModelBoxDirInode::ModelBoxDirInode() {
  SetInodeType(MODELBOX_FUSE_INODE_TYPE_DIR);
};

ModelBoxDirInode::~ModelBoxDirInode() = default;

void ModelBoxDirInode::SetStat(struct stat *stat) {
  stat_ = *stat;
  stat_.st_mode |= S_IFDIR;
}

int ModelBoxDirInode::FillStat(struct stat *stat) {
  *stat = stat_;
  auto dentry = GetDEntry();
  if (dentry) {
    stat->st_nlink += dentry->ChildDirNum();
  }
  return 0;
}

ModelBoxFileInode::ModelBoxFileInode() {
  SetInodeType(MODELBOX_FUSE_INODE_TYPE_FILE);
}
ModelBoxFileInode::~ModelBoxFileInode() = default;

ModelBoxDEntry::ModelBoxDEntry() = default;
ModelBoxDEntry::~ModelBoxDEntry() = default;

void ModelBoxDEntry::SetParent(const std::shared_ptr<ModelBoxDEntry> &dentry) {
  parent_ = dentry;
}

std::shared_ptr<ModelBoxDEntry> ModelBoxDEntry::LookUp(
    const std::string &path) {
  auto split_path = SplitPath(path, '/');
  if (path == "/") {
    return shared_from_this();
  }

  if (split_path.size() > 0 && split_path.front().length() == 0) {
    split_path.pop_front();
  }

  return LookUp(split_path);
}

std::shared_ptr<ModelBoxDEntry> ModelBoxDEntry::LookUp(
    std::list<std::string> &names) {
  if (names.size() <= 0) {
    return shared_from_this();
  }

  auto &current = names.front();
  names.pop_front();
  if (current == ".") {
    return shared_from_this();
  }

  if (current == "..") {
    return Parent();
  }

  std::unique_lock<std::mutex> lock(children_lock_);
  auto itr = children_.find(current);
  if (itr == children_.end()) {
    return nullptr;
  }
  auto cur_entry = itr->second;
  lock.unlock();

  return cur_entry->LookUp(names);
}

int ModelBoxDEntry::AddChild(const std::shared_ptr<ModelBoxDEntry> &dentry) {
  const auto &name = dentry->GetName();
  std::unique_lock<std::mutex> lock(children_lock_);
  auto itr = children_.find(name);
  if (itr != children_.end()) {
    return -1;
  }

  if (dentry->inode_->GetInodeType() == MODELBOX_FUSE_INODE_TYPE_DIR) {
    dir_num_++;
  }
  children_[name] = dentry;
  return 0;
}

int ModelBoxDEntry::RmvChild(const std::string &name) {
  std::unique_lock<std::mutex> lock(children_lock_);
  auto itr = children_.find(name);
  if (itr == children_.end()) {
    return -1;
  }

  if (itr->second->inode_->GetInodeType() == MODELBOX_FUSE_INODE_TYPE_DIR) {
    dir_num_--;
  }

  children_.erase(name);
  return 0;
}

int ModelBoxDEntry::ChildNum() {
  std::unique_lock<std::mutex> lock(children_lock_);
  return children_.size();
}

int ModelBoxDEntry::ChildDirNum() {
  std::unique_lock<std::mutex> lock(children_lock_);
  return dir_num_;
}

void ModelBoxDEntry::SetName(const std::string &name) { name_ = name; }

const std::string &ModelBoxDEntry::GetName() { return name_; }

void ModelBoxDEntry::SetInode(const std::shared_ptr<ModelBoxInode> &inode) {
  inode_ = inode;
  inode_->SetDEntry(shared_from_this());
}

std::shared_ptr<ModelBoxInode> ModelBoxDEntry::GetInode() { return inode_; }

std::shared_ptr<ModelBoxDEntry> ModelBoxDEntry::Parent() {
  return parent_.lock();
}
std::vector<std::shared_ptr<ModelBoxDEntry>> ModelBoxDEntry::Children() {
  std::vector<std::shared_ptr<ModelBoxDEntry>> result;
  std::unique_lock<std::mutex> lock(children_lock_);
  for (auto &item : children_) {
    result.push_back(item.second);
  }
  return result;
}

}  // namespace modelbox