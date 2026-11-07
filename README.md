# OpenViking SeekDB Adapter

为 [OpenViking](https://github.com/volcengine/OpenViking) `v0.5.3` 提供
[OceanBase SeekDB](https://github.com/oceanbase/seekdb) 集群级向量后端。

插件通过 SeekDB 官方 MySQL 协议直接使用：

- `VECTOR` + HNSW/HNSW_SQ/HNSW_BQ/IVF/IVF_PQ，或无索引 exact scan：dense vector
- `SPARSEVECTOR` + SINDI：OpenViking learned sparse vector
- `FULLTEXT INDEX` + `MATCH ... AGAINST`：全文/BM25 风格召回
- JSON scalar metadata：OpenViking 的 eq/in/range/contains/path-depth 过滤 DSL

## 已实现的能力

- OpenViking 动态类路径加载，无需修改后端枚举
- 集合、索引、描述和数据生命周期
- `upsert_data` 全量替换语义
- `update_data` 真正部分更新，未提交字段保持不变
- 按主键批量写入、读取、删除，并保持 `fetch_data` 的请求顺序
- dense、sparse 与双路加权融合检索
- 全文检索，保留 `content` 字段
- `count`、分组过滤计数和稳定的 scalar 排序
- URI 前缀及深度过滤
- account/ACL 字段普通过滤，不绕过上层租户策略
- 可配置连接池、超时、SSL、HNSW 参数、稀疏向量上限和批量大小
- PyMySQL 连接失效检测与重连

## 兼容性

- OpenViking：`0.5.3`（接口按该版本实现）
- Python：`3.10` 到 `3.14`
- SeekDB server：建议 `>= 1.4.0`。dense 检索所需版本较低，但 `SPARSEVECTOR` 与全文能力建议使用 1.4.x
- 连接模式：SeekDB Server/OceanBase MySQL 协议；不支持 embedded `pyseekdb.Client(path=...)`

当前集成测试在 SeekDB `1.4.0-100000172026082615` 上覆盖 cosine、L2、
inner product、flat、HNSW、HNSW_SQ、HNSW_BQ、IVF_FLAT、IVF_PQ、sparse、
hybrid 和 full-text 检索。

OpenViking `0.5.3` 把关键词与 grep 的支持后端硬编码为
`volcengine`、`vikingdb`、`opengauss`。插件采用兼容模式向 OpenViking
报告 `opengauss`，使现有关键词请求可以直接进入 SeekDB 的全文索引；插件自身
仍通过 `backend_name="seekdb"` 和 `capabilities` 标识真实实现。

如果设置 `openviking_fulltext_compat=false`，插件会报告原生 `seekdb` 模式。
此时 dense/sparse 检索可用，但 OpenViking `0.5.3` 会在请求边界拒绝 keyword
搜索，grep 也会退回文件扫描。除非上游允许列表已经加入 `seekdb`，否则请保持
默认值。

## 安装

插件必须安装到 OpenViking 实际使用的 Python 环境。使用 `uv pip --python` 指定该
解释器，不需要手工激活虚拟环境：

```bash
cd /path/to/openviking-seekdb
OPENVIKING_PYTHON=/path/to/openviking/.venv/bin/python
uv pip install --python "$OPENVIKING_PYTHON" .
```

确认动态类路径可加载：

```bash
"$OPENVIKING_PYTHON" - <<'PY'
from openviking_seekdb import SeekDBCollectionAdapter
print(SeekDBCollectionAdapter)
PY
```

插件只新增 Python 包，不会自动切换现有配置，也不会迁移本地 `local` 后端数据。

本地开发和测试使用独立的 uv 虚拟环境。仓库中的 `.python-version` 固定开发环境为
Python 3.13；`uv sync` 会据此创建 `.venv`：

```bash
cd /path/to/openviking-seekdb
uv sync --extra test
uv run --extra test python - <<'PY'
from openviking_seekdb import SeekDBCollectionAdapter
print(SeekDBCollectionAdapter)
PY
```

依赖发生变化后使用 `uv lock` 更新 `uv.lock`。需要重新创建开发环境时，可以删除
`.venv` 后再次执行 `uv sync --extra test`，不需要手工安装 `pip`。

## 配置

下面是最小示例。完整文件见
[`examples/ov.conf.example.json`](examples/ov.conf.example.json)。

```json
{
  "storage": {
    "vectordb": {
      "backend": "openviking_seekdb.adapter.SeekDBCollectionAdapter",
      "name": "context",
      "index_name": "default",
      "dimension": 1024,
      "distance_metric": "cosine",
      "sparse_weight": 0.3,
      "custom_params": {
        "host": "127.0.0.1",
        "port": 2881,
        "database": "openviking",
        "user": "root",
        "tenant": null,
        "password_env": "SEEKDB_PASSWORD",
        "pool_size": 8,
        "vector_index_type": "hnsw",
        "fulltext": true,
        "fulltext_analyzer": "ik",
        "sparse_max_dimension": 500000,
        "candidate_limit": 1000
      }
    }
  }
}
```

`dimension` 必须与 OpenViking embedding 配置一致。`sparse_weight` 为 `0` 时不创建
sparse 索引；使用 learned sparse embedding 时设置为 `(0, 1]`。

密码不会写入仓库。`password` 可直接出现在运行时配置中，也可用默认的
`SEEKDB_PASSWORD`，或通过 `password_env` 指定其他环境变量。日志不会输出密码。

### 向量索引类型

HNSW 不是 OpenViking 的固定要求。OpenViking 的核心索引元数据允许不同
`IndexType`：默认 `local` 后端使用 `flat`/`flat_hybrid`（索引元数据默认
`Quant=int8`），内置 openGauss 适配器默认使用 `hnsw`，Volcengine/VikingDB
适配器默认使用 `hnsw`/`hnsw_hybrid`。本插件为了保持既有行为默认使用 SeekDB
`hnsw`，可通过 `custom_params.vector_index_type` 选择：

| `vector_index_type` | SeekDB 实现 | 量化/存储特性 | 距离限制 |
|---|---|---|---|
| `flat` | 不创建 dense ANN 索引，SQL 精确全扫描 | 召回最高、规模增大后延迟和资源开销高 | cosine/l2/ip |
| `hnsw` | HNSW | 内存图索引，无额外量化 | cosine/l2/ip |
| `hnsw_sq` | HNSW_SQ | 标量量化，降低索引内存占用 | cosine/l2/ip |
| `hnsw_bq` | HNSW_BQ | 二值量化，进一步压缩索引；可配置重排 | cosine/l2 |
| `ivf_flat` | IVF_FLAT | 磁盘型 IVF 聚类索引 | cosine/l2/ip |
| `ivf_pq` | IVF_PQ | IVF + product quantization | cosine/l2/ip |

`vector_index_type` 只在 collection 首次创建索引时生效。已有 collection 修改该
配置不会原地换索引；需要删除 collection 后重建，或在数据库侧使用 SeekDB 的索引
重建工具。

例如使用 HNSW 标量量化：

```json
"custom_params": {
  "vector_index_type": "hnsw_sq",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200,
  "hnsw_ef_search": 64
}
```

使用 HNSW 二值量化：

```json
"custom_params": {
  "vector_index_type": "hnsw_bq",
  "hnsw_m": 16,
  "hnsw_ef_construction": 200,
  "hnsw_ef_search": 64,
  "hnsw_refine_k": 4.0,
  "hnsw_refine_type": "sq8",
  "hnsw_bq_bits_query": 32,
  "hnsw_bq_use_fht": true
}
```

使用 IVF_PQ 时，`ivf_pq_m` 必须整除向量维度；省略时会选择不大于 8 的最大可行值：

```json
"custom_params": {
  "vector_index_type": "ivf_pq",
  "ivf_nlist": 128,
  "ivf_sample_per_nlist": 256,
  "ivf_nbits": 8,
  "ivf_pq_m": 16,
  "ivf_nprobes": 8
}
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `host` | `127.0.0.1` | SeekDB 主机 |
| `port` | `2881` | SeekDB SQL 端口 |
| `database` | `openviking` | 必须预先存在 |
| `user` / `tenant` | `root` / 空 | 同时填写 tenant 时生成 `user@tenant` |
| `pool_size` | `8` | 每 adapter 的最大连接数 |
| `connect_retries` | `3` | 首次建连或失效重连的最大尝试次数 |
| `vector_index_type` | `hnsw` | `flat`、`hnsw`、`hnsw_sq`、`hnsw_bq`、`ivf_flat`、`ivf_pq` |
| `fulltext` | `true` | 是否创建 content 全文索引 |
| `fulltext_analyzer` | `ik` | `ik`、`space`、`beng`、`ngram`、`ngram2` |
| `hnsw_m` | `16` | HNSW 邻居上限 |
| `hnsw_ef_construction` | `200` | 构建候选数 |
| `hnsw_ef_search` | `64` | 查询候选数 |
| `hnsw_extra_info_max_size` | 未设置 | HNSW 主键附加信息上限；部分 SeekDB 版本会拒绝该选项 |
| `hnsw_refine_k` | SeekDB 默认值 | 仅 HNSW_BQ：重排候选比例 |
| `hnsw_refine_type` | SeekDB 默认值 | 仅 HNSW_BQ：`sq8` 或 `fp32` |
| `hnsw_bq_bits_query` | SeekDB 默认值 | 仅 HNSW_BQ：`0`、`4` 或 `32` |
| `hnsw_bq_use_fht` | SeekDB 默认值 | 仅 HNSW_BQ：是否启用 FHT |
| `ivf_nlist` | `128` | IVF 聚类中心数 |
| `ivf_sample_per_nlist` | `256` | IVF 每个聚类中心的后建采样数 |
| `ivf_nbits` | `8` | IVF_PQ 每个子向量的量化位数 |
| `ivf_pq_m` | 自动 | IVF_PQ 子量化器数量 |
| `ivf_nprobes` | SeekDB 会话默认值 | IVF 查询访问的聚类中心数，不得大于 `ivf_nlist` |
| `candidate_limit` | `1000` | hybrid/time-decay 每路候选数 |
| `sparse_max_dimension` | `500000` | learned sparse term 的整数空间 |
| `sparse_max_nonzero` | `4096` | 单条稀疏向量最大非零项 |
| `refresh_after_write` | `true` | 写入后调用 SeekDB 1.3+ 的索引刷新；由外部统一刷新时可关闭 |
| `openviking_fulltext_compat` | `true` | 使用 `opengauss` 模式标签通过 v0.5.3 允许列表 |

`custom_params` 也接受嵌套的 `"seekdb": { ... }`，嵌套对象优先。

## 数据布局

每个 OpenViking collection 对应一张 SeekDB HEAP 表：

```text
id               VARCHAR(512) PK
metadata         JSON          # OpenViking scalar fields
content          LONGTEXT      # FULLTEXT INDEX
embedding        VECTOR(N)     # flat 或可配置 ANN 索引
sparse_embedding SPARSEVECTOR  # SINDI, optional
sparse_terms     JSON          # 原始 learned-sparse term -> weight
```

集合目录和索引目录存放在：

```text
_ov_seekdb_collections
_ov_seekdb_indexes
```

前缀可由 `metadata_prefix` 和 `table_prefix` 调整。逻辑 collection/index 名称会转换为
确定性的物理名称，避免 OpenViking 名称超过 SeekDB 标识符限制。

## 混合检索

OpenViking 0.5.3 的公开搜索类型只有 `semantic` 和 `keywords`，没有单独名为
`hybrid` 的 `SearchType`。当 embedding 配置为 `hybrid`，或同时生成 dense 和
sparse 向量时，`semantic` 请求会把 `dense_vector` 与 `sparse_vector` 一起传给
`Collection.search_by_vector`。本插件据此分别查询 HNSW/IVF dense 索引与 SINDI
sparse 索引，再用 `sparse_weight` 做加权融合；因此 OpenViking 的 dense+sparse
hybrid 通路已经可用。

SeekDB 还提供另一类混合检索：`DBMS_HYBRID_SEARCH.SEARCH` 可以把全文/BM25 查询与
dense 向量查询合并，并支持 weighted 或 RRF 排名。OpenViking 0.5.3 的通用
`CollectionAdapter.query` 只接收向量，不接收原始查询文本，因此这个全文+dense
RRF 接口无法通过现有通用适配器契约自动接入。当前插件提供的是 OpenViking 可调用的
dense+sparse hybrid，以及独立的 `search_by_keywords` 全文召回；两者不会在插件内
自动再做 RRF 融合。

## 语义说明

- `upsert_data` 覆盖整条记录；省略的标量字段不会继承旧值。
- `update_data` 在事务内使用 `SELECT ... FOR UPDATE`，只替换显式提交的字段。
- `fetch_data` 按输入 ID 顺序返回，并报告 `ids_not_exist`。
- vector 分数统一为越高越好：cosine 使用 `1-distance`，L2 使用
  `1/(1+distance)`，inner product 直接使用点积。
- URI 由 OpenViking adapter 基类在写入时编码成 `/...`，读取时恢复成 `viking://...`。
- `PathScope.depth=0` 是精确匹配；`depth<0` 是无界子树；正数是相对层级上限。
- scalar 排序始终追加 `id` 作为稳定次序，删除分页不会因自身删除而漂移。
- SeekDB 原生 sparse vector 使用整数 key。任意 learned-sparse 字符串 key 会被
  稳定哈希到 `1..sparse_max_dimension`；同一条记录内发生碰撞时权重相加。
  原始 key/value 保留在 `sparse_terms`，因此 fetch 不丢失原始表示。
- 当前不实现 TTL 和多模态检索；传入非零 TTL 或调用多模态接口会明确报错。

## 限制

- scalar 字段当前使用 JSON 通用过滤，未为每个字段建立独立二级索引。更大规模数据上
  应优先保留 OpenViking 顶层 `account_id`/URI scope 过滤；后续可在
  `schema.py` 和 `collection.py` 增加 generated columns 或 functional indexes。
- learned sparse 的字符串 term 映射是确定性的，但这意味着直接与另一套
  `sparse_max_dimension` 配置写入的数据不可混用；扩容词表空间应重建 sparse 索引。
- `distance_metric=ip` 的 dense APPROXIMATE 查询顺序按 SeekDB 当前 SQL 语法生成；
  上线前应使用实际 SeekDB 版本做小规模 recall 验证。
- 插件不迁移旧 `local` 集合。切换后 OpenViking 会创建新的 SeekDB collection。

## 测试

```bash
uv sync --extra test
uv run --extra test pytest
uv run --extra test ruff check .
uv run --extra test ruff format --check .
```

真实集成测试使用环境变量指向临时 SeekDB：

```bash
SEEKDB_TEST_HOST=127.0.0.1 \
SEEKDB_TEST_PORT=2881 \
SEEKDB_TEST_DATABASE=openviking_test \
SEEKDB_TEST_USER=root \
uv run --extra test pytest -m integration
```

测试会创建名称带随机后缀的临时 collection，并在结束时删除。

## 容器镜像

仓库提供[`Dockerfile`](Dockerfile)，可基于任意 OpenViking `v0.5.3` 镜像安装插件：

```bash
podman build \
  --build-arg OPENVIKING_IMAGE=ghcr.io/volcengine/openviking:0.5.3 \
  -t openviking-seekdb:0.5.3 .
```

如果基础镜像没有 `python`/`python3` 指向 OpenViking 解释器，请通过
`PYTHON_BIN` build arg 指向正确路径。

## 官方依据

- SeekDB HNSW：https://docs.seekdb.ai/hnsw-index
- SeekDB IVF：https://docs.seekdb.ai/ivf-index
- SeekDB sparse vector：https://docs.seekdb.ai/in-memory-sparse-vector-index
- SeekDB full-text：https://docs.seekdb.ai/full-text-search
- SeekDB hybrid search：https://docs.seekdb.ai/hybrid-search
- SeekDB PyMySQL：https://docs.seekdb.ai/pymysql-connection-to-seekdb-sample-program
- OpenViking：https://github.com/volcengine/OpenViking
