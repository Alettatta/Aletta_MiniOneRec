import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import datetime
import torch
from tqdm import tqdm
import numpy as np


def clean_text(text):
    """Clean text by removing HTML tags and excessive whitespace"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = html.unescape(text)
    # Replace quotes
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_path(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """Write data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """Write index mapping to file"""
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


# Dataset name mapping
amazon18_dataset2fullname = {
    'Arts': 'Arts_Crafts_and_Sewing',
    'Games': 'Video_Games',
    'Sports': 'Sports_and_Outdoors',
    'Instruments': 'Musical_Instruments',
    'Scientific': 'Scientific'
}


def get_timestamp_start(year, month):
    """Get timestamp for the start of a given year and month"""
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def load_metadata_json2csv_style(category, metadata_file=None):
    """Load metadata using json2csv style processing"""
    if metadata_file is None:
        metadata_file = f'../meta_{category}.json'
    
    metadata = []
    try:
        with open(metadata_file) as f:
            metadata = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Metadata file {metadata_file} not found")
        return [], {}, set()
    
    id_title = {}
    remove_items = set()
    
    for meta in tqdm(metadata, desc="Processing metadata"):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        
        # Clean title like json2csv
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        
        if len(meta['title']) > 1 and len(meta['title'].split(" ")) <= 20:
            id_title[meta['asin']] = meta['title']
            # Store full metadata for later use
        else:
            remove_items.add(meta['asin'])
    
    return metadata, id_title, remove_items


def load_reviews_json2csv_style(category, reviews_file=None, start_timestamp=None, end_timestamp=None):
    """Load reviews using json2csv style processing"""
    if reviews_file is None:
        try:
            with open(f'../{category}_5.json') as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            try:
                with open(f'../{category}.json') as f:
                    reviews = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Reviews file not found for category {category}")
                return []
    else:
        try:
            with open(reviews_file) as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Reviews file {reviews_file} not found")
            return []
    
    # NOTE: Don't filter by timestamp here, do it in k-core filtering like json2csv
    return reviews


def k_core_filtering_json2csv_style(reviews, id_title, user_k=5, item_k=5, start_timestamp=None, end_timestamp=None):
    """Perform k-core filtering using json2csv style logic"""
    # 建立两个黑名单：以后看到这些用户和item就跳过
    remove_users = set()
    remove_items = set()
    
    # 删除没有metadata的商品（因为Item embedding必须要Title+description）
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])
    
    # Iterative k-core filtering (exactly like json2csv)
    while True: # 不断过滤直到稳定
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict() # 每个user多少条review
        item_counts = dict() # 每个item多少人交互
        
        for review in tqdm(reviews, desc="K-core filtering"):
            # Filter by timestamp INSIDE the loop like json2csv
            if start_timestamp and end_timestamp: # 按照时间过滤
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue
            
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue
            
            if review['reviewerID'] not in user_counts:
                user_counts[review['reviewerID']] = 0
            user_counts[review['reviewerID']] += 1
            
            if review['asin'] not in item_counts:
                item_counts[review['asin']] = 0
            item_counts[review['asin']] += 1
            
            total += 1
            new_reviews.append(review)
        
        # Mark users/items for removal if below threshold
        for user in user_counts:
            if user_counts[user] < user_k:
                remove_users.add(user)
                flag = True
        
        for item in item_counts:
            if item_counts[item] < item_k:
                remove_items.add(item)
                flag = True

        # density = review数/(用户数*商品数)，防止太过稀疏
        print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}, Density: {total / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
        
        if not flag:
            break # 如果没有任何user和item需要删除，说明已经稳定，可以结束while True循环
        
        reviews = new_reviews
    
    return new_reviews, user_counts, item_counts


# 输入review(reviewerID, asin, overall, unixReviewTime)
# 输出用户序列(user2items)、用户编号(user2index)、商品编号(item2index)、交互表四元组(interactions, 包含user, item, rating, timestamp)
# user2items：{0:[0, 1, 2]}；user2index：{U1: 0}；item2index：{A:0}；interactions：[(U1,A,5,100),(U1,B,4,200),(U2,C,5,150)]
def convert_inters2dict_amazon18_style(reviews):
    """Convert interactions to dict format like amazon18_data_process"""
    user2items = collections.defaultdict(list) # 例如user0:[商品0, 3, 5]
    user2index, item2index = dict(), dict() # 把原始 user ID / ASIN 重编号为整数 ID
    
    # Sort reviews by timestamp for each user
    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)
    
    # Sort each user's reviews by timestamp (don't remove duplicates like json2csv)
    # 注意上面这个注释：只排序，没有去重。也就是说，如果同一个用户多次评论或购买了同一个商品，要保留多次记录，例如U1: A, B, A, C
    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))
    
    # Create mappings and interactions
    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        
        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)
            
            user_items.append(item)
            interactions.append((
                user, item, 
                float(review['overall']), 
                int(review['unixReviewTime'])
            ))
        
        user2items[user2index[user]] = [item2index[item] for item in user_items]
    
    return user2items, user2index, item2index, interactions


# 把逐条review转换为“用户历史商品序列 → 下一个目标商品”的监督学习样本
def generate_interaction_list_json2csv_style(reviews, user2index, item2index, id_title):
    """Generate interaction list like json2csv for 8:1:1 split"""
    # Create user interactions similar to json2csv
    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}
    
    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'item_ids': [],
                'titles': []
            }
        
        # 以用户为单位收集每条 review 的原始 ASIN、整数 item ID、评分、时间戳、标题
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])
        interact[user]['titles'].append(id_title[item])
    
    # 对每位用户的所有交互按 unixReviewTime 升序排序
    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']
        
        # Sort all by timestamp
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        items, ratings, timestamps, item_ids, titles = list(items), list(ratings), list(timestamps), list(item_ids), list(titles)
        
        # 从该用户的第二次交互开始构造样本，历史窗口最多保留最近 10 个商品（一个用户如果仅保留 1 次交互，不能产生任何序列样本）
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([
                user,                    # user_id
                items[st:i],            # item_asins (history)
                items[i],               # item_asin (target)
                item_ids[st:i],         # history_item_id
                item_ids[i],            # item_id (target)
                titles[st:i],           # history_item_title
                titles[i],              # item_title (target)
                ratings[st:i],          # history_rating
                ratings[i],             # rating (target)
                timestamps[st:i],       # history_timestamp
                timestamps[i]           # timestamp (target)
            ])
    
    # 最后把所有用户生成的样本，按目标商品的交互时间全局排序
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list 
    # 这个list包含：user 原始 ID、历史原始 ASIN 列表、目标原始 ASIN、历史内部 item-id 列表、目标内部 item-id、
    # 历史商品标题列表,目标商品标题,历史评分列表,目标评分,历史时间戳列表,目标时间戳


# 将上一个函数得到、且已按时间全局排序的序列样本，按 8:1:1 切分并写入 train、valid、test
# inter文件每行格式为：user_id:token    item_id_list:token_seq    item_id:token
# 例如 3545（用户）    245 781 1003（历史item序列）    219（目标item-id）
def convert_to_atomic_files_json2csv_style(args, interaction_list, user2index):
    """Convert interaction list to train/valid/test files using 8:1:1 split like json2csv"""
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    
    # Create output directories
    check_path(os.path.join(args.output_path, args.dataset))
    
    # Split 8:1:1 like json2csv
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)
    
    train_interactions = interaction_list[:train_end]
    valid_interactions = interaction_list[train_end:valid_end]
    test_interactions = interaction_list[valid_end:]
    
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Valid interactions: {len(valid_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    # Write train file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in train_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write valid file  
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in valid_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write test file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in test_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original] 
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids  
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    return train_interactions, valid_interactions, test_interactions


# 为保留的交互保存评论正文和摘要，生成 Industrial_and_Scientific.review.json
# 形如{"(uid, iid, timestamp)": {"review": "...", "summary": "..."}}
def load_review_data_amazon18_style(reviews, user2index, item2index):
    """Load review data like amazon18_data_process"""
    review_data = {}
    
    for review in tqdm(reviews, desc='Load reviews'):
        try:
            user = review['reviewerID']
            item = review['asin']
            
            if user in user2index and item in item2index:
                uid = user2index[user]
                iid = item2index[item]
                
                # Use timestamp to create unique keys for same user-item pairs at different times
                timestamp = review['unixReviewTime']
                unique_key = str((uid, iid, timestamp))
                
            else:
                continue
                
            if 'reviewText' in review:
                review_text = clean_text(review['reviewText'])
            else:
                review_text = ''
                
            if 'summary' in review:
                summary = clean_text(review['summary'])
            else:
                summary = ''
                
            review_data[unique_key] = {"review": review_text, "summary": summary}
            
        except (ValueError, KeyError):
            continue
    
    return review_data


# 为最终保留的商品构造统一 metadata，生成后续 item text embedding 所需的 Industrial_and_Scientific.item.json
# {
#     0: {
#         "title": "...", 
#         "description": "...",
#         "brand": "...",
#         "categories": "..."
#     },
#     ...
# }
# 后续：item.json → 读取每个 item 的 title + description → Qwen 文本编码 → .emb-qwen-td.npy
def create_item_features_amazon18_style(metadata, item2index, id_title):
    """Create item features like amazon18_data_process"""
    item2feature = collections.defaultdict(dict)
    
    # Create a mapping from asin to metadata
    asin_to_meta = {}
    for meta in metadata:
        asin_to_meta[meta['asin']] = meta
    
    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            
            title = id_title.get(item_asin, clean_text(meta.get("title", "")))
            
            descriptions = meta.get("description", "")
            if descriptions:
                descriptions = clean_text(descriptions)
            else:
                descriptions = ""
                
            brand = meta.get("brand", "").replace("by\n", "").strip()
            
            categories = meta.get("categories", meta.get("category", []))
            if categories and len(categories) > 0:
                # Handle both list and string formats
                if isinstance(categories[0], list):
                    # Flatten nested categories
                    flat_categories = []
                    for cat_group in categories:
                        flat_categories.extend(cat_group)
                    categories = flat_categories
                
                # Filter out invalid categories
                new_categories = []
                for category in categories:
                    if "</span>" not in str(category):
                        new_categories.append(str(category).strip())
                categories = ",".join(new_categories).strip()
            else:
                categories = ""
            
            item2feature[item_id] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories
            }
    
    return item2feature


# 读取 metadata → k-core 筛选 → 检查最终商品数 → 必要时向前扩展时间窗口并重新处理
# 成功时返回五元组
# (
#     filtered_reviews,
#     user_counts,
#     item_counts,
#     metadata,
#     id_title
# )
def process_dataset_recursive(args, metadata, reviews, start_timestamp, end_timestamp):
    """Process dataset with recursive year reduction like json2csv"""
    
    # Load metadata 
    metadata, id_title, remove_items = load_metadata_json2csv_style(
        args.dataset, args.metadata_file
    )
    
    if not metadata:
        print(f"Error: No metadata found for dataset {args.dataset}")
        return None
    
    print(f"Loaded {len(metadata)} metadata items, {len(id_title)} with valid titles")
    
    # Perform k-core filtering with timestamp filtering inside
    print("Performing k-core filtering...")
    filtered_reviews, user_counts, item_counts = k_core_filtering_json2csv_style(
        reviews, id_title, args.user_k, args.item_k, start_timestamp, end_timestamp
    )
    
    print(f"After filtering: {len(user_counts)} users, {len(item_counts)} items, {len(filtered_reviews)} reviews")
    
    # 若不足 3000 个商品，将起始年份向前移动一年，再递归重新处理
    if args.st_year > 1996 and len(item_counts) < 3000:
        print(f"Items count {len(item_counts)} < 3000, expanding time range...")
        args.st_year -= 1
        new_start_timestamp = get_timestamp_start(args.st_year, args.st_month)
        print(f"New time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}")
        return process_dataset_recursive(args, metadata, reviews, new_start_timestamp, end_timestamp)
    
    return filtered_reviews, user_counts, item_counts, metadata, id_title


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games / Sports')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=10, help='start month')
    parser.add_argument('--ed_year', type=int, default=2018, help='end year')
    parser.add_argument('--ed_month', type=int, default=11, help='end month')
    parser.add_argument('--metadata_file', type=str, default=None, help='metadata file path')
    parser.add_argument('--reviews_file', type=str, default=None, help='reviews file path')
    parser.add_argument('--output_path', type=str, default='./data', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Processing dataset: {args.dataset}')
    print(f'Initial time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}')
    print(f'K-core thresholds: user_k={args.user_k}, item_k={args.item_k}')
    
    # Set time range
    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)
    
    # Load reviews first (without timestamp filtering)
    print("Loading reviews...")
    reviews = load_reviews_json2csv_style(
        args.dataset, args.reviews_file
    )
    
    if not reviews:
        print(f"Error: No reviews found for dataset {args.dataset}")
        print("Please check if the reviews file exists and try again.")
        exit(1)
        
    print(f"Loaded {len(reviews)} total reviews")
    
    # Process dataset with recursive logic
    result = process_dataset_recursive(args, None, reviews, start_timestamp, end_timestamp)
    
    if result is None:
        print("Failed to process dataset")
        exit(1)
    
    filtered_reviews, user_counts, item_counts, metadata, id_title = result
    
    print(f"Final filtering results:")
    print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(filtered_reviews)}")
    print(f"Density: {len(filtered_reviews) / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
    
    # Convert to amazon18 style format
    print("Converting to amazon18 format...")
    user2items, user2index, item2index, interactions = convert_inters2dict_amazon18_style(filtered_reviews)
    
    print(f"After amazon18 conversion:")
    print(f"  User2index: {len(user2index)} users")
    print(f"  Item2index: {len(item2index)} items") 
    print(f"  Interactions: {len(interactions)} interactions")
    
    # Generate interaction list for 8:1:1 split like json2csv
    print("Generating interaction list for 8:1:1 split...")
    interaction_list = generate_interaction_list_json2csv_style(
        filtered_reviews, user2index, item2index, id_title
    )
    
    print(f"Generated {len(interaction_list)} interaction sequences")
    
    # Create output directory and split data
    train_interactions, valid_interactions, test_interactions = convert_to_atomic_files_json2csv_style(
        args, interaction_list, user2index
    )
    
    # Generate user2items for compatibility with amazon18 format
    user2items_final = collections.defaultdict(list)
    for user_idx, item_list in user2items.items():
        user2items_final[user_idx] = item_list
    
    # Write interaction files (amazon18 style output)
    write_json_file(user2items_final, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
    
    # Create item features
    print("Creating item features...")
    item2feature = create_item_features_amazon18_style(metadata, item2index, id_title)
    
    # Load review data
    print("Loading review data...")
    review_data = load_review_data_amazon18_style(filtered_reviews, user2index, item2index)
    
    print(f"Final statistics:")
    print(f"Users: {len(user2index)}")
    print(f"Items: {len(item2index)}")
    print(f"Reviews: {len(review_data)}")
    print(f"Total interaction sequences: {len(interaction_list)}")
    print(f"Train sequences: {len(train_interactions)}")
    print(f"Valid sequences: {len(valid_interactions)}")
    print(f"Test sequences: {len(test_interactions)}")
    
    # Write output files (amazon18 style)
    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    write_json_file(review_data, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))
    
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))
    
    print("Processing completed!")
