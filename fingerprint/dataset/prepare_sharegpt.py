import os
import json


download_dir = "datasets/user/sharegpt/"

if not os.path.exists(download_dir + 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'):
    os.system(f"wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json -P {download_dir}")

with open(download_dir + 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)


with open(download_dir + 'sharegpt.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for item in data:
        conversation_id = item['id']
        conversations = item['conversations']
        
    
        formatted_conversation = []
    
        i = 0
        while i < len(conversations):
            if conversations[i]['from'] == 'gpt':   # gpt
                formatted_conversation.append({
                    "assistant": conversations[i]['value']
                })
                i += 1
            else:   # human
                if i + 1 < len(conversations): # has next
                    if conversations[i+1]['from'] == 'gpt': # next is gpt
                        formatted_conversation.append({
                            "human": conversations[i]['value'],
                            "assistant": conversations[i+1]['value']
                        })
                        i += 2
                    else:   # next is human
                        formatted_conversation.append({
                            "human": conversations[i]['value']
                        })
                        i += 1
                    # assert conversations[i+1]['from'] == 'gpt', f"{conversations}"
                    # formatted_conversation.append({
                    #     "human": conversations[i]['value'],
                    #     "assistant": conversations[i+1]['value']
                    # })
                else:   # no next
                    formatted_conversation.append({
                        "human": conversations[i]['value']
                    })
                    i += 1
          
        new_entry = {
            "conversation_id": conversation_id,
            "conversation": formatted_conversation,
            "dataset": "sharegpt"
        }
        
        jsonl_file.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
