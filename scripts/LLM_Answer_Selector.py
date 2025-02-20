import pandas as pd
import openai
from tqdm import tqdm
import os

def load_api_key(filename="openai_api_key.txt"):
    """Load API key from file"""
    with open(filename) as file:
        return file.read().strip()

def get_llm_response(client, dialogue):
    """Get response from LLM"""
    prompt = f"""
你是一個聊天機器人：
使用者回覆你：{dialogue}
請從以上選項中選出一個最適合的回覆，回覆對方。
僅用一個字母（A、B、C、D、E 或 F）回答：
請僅輸出一個字母作為您的答案。
"""
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=0.3
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 發生錯誤: {e}")
        exit(1)

def save_answer(answer):
    """Save answer to CSV file with utf-8-sig encoding"""
    answer_df = pd.DataFrame([answer])
    if not os.path.exists('ans.csv'):
        answer_df.to_csv('ans.csv', index=False, encoding='utf-8-sig')
    else:
        answer_df.to_csv('ans.csv', mode='a', header=False, index=False, encoding='utf-8-sig')

def main():
    # Read predictions.csv
    df = pd.read_csv('predictions.csv')
    
    # Initialize OpenAI client
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key)
    
    # Process each dialogue with progress bar
    valid_choices = {'A', 'B', 'C', 'D', 'E', 'F'}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        # Format options
        options = {
            'A': row['label_1'],
            'B': row['label_2'],
            'C': row['label_3'],
            'D': row['label_4'],
            'E': row['label_5'],
            'F': "以上選項皆不符合"
        }
        
        # Get LLM response
        dialogue = row['title']
        response = get_llm_response(client, dialogue + "\n" + "\n".join([f"{k}. {v}" for k, v in options.items()]))
        
        # Map response to actual answer or mark as error
        answer = options.get(response, "錯誤")
        
        # Prepare the answer data
        answer_data = {'title': row['title'], 'answer': answer}
        
        # Save each answer to CSV after processing each dialogue
        save_answer(answer_data)
        
        # Add delay to respect rate limits

if __name__ == "__main__":
    main()