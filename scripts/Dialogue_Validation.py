import pandas as pd
import openai
from tqdm import tqdm
import os
import time

def load_api_key(filename="openai_api_key.txt"):
    """Load API key from file"""
    with open(filename) as file:
        return file.read().strip()

def parse_llm_response(response):
    """Parse LLM response to extract answers for each question"""
    try:
        # Split the response by commas and clean up
        answers = [ans.strip() for ans in response.split(',') if ans.strip()]
        
        # Extract just the "成立" or "不成立" part for each answer
        parsed_answers = []
        for answer in answers:
            if "：" in answer:
                # Get the part after the colon
                result = answer.split("：")[1].strip()
            else:
                # If no colon, just get the answer directly
                result = answer.strip()
            
            if "成立" in result:
                parsed_answers.append(result)
        
        return parsed_answers
    except Exception as e:
        print(f"解析回應時發生錯誤: {e}")
        return ["解析錯誤"] * 10

def get_llm_response(client, dialogues):
    """Get response from LLM"""
    prompt = """你是一名嚴厲的中文老師。 判斷情境題：以下十題題目，一步一步思考，判斷他們的對話是否成立。成立的話回答成立，否則回答不成立，每題以,分隔，不要回答無關的答案

輸出格式：
題號：答案
範例如下：
1：成立,
2：不成立,
...

題目：
"""
    # Add all 10 dialogues to the prompt
    for i, dialogue in enumerate(dialogues, 1):
        prompt += f"({i})\n"
        prompt += f"小明：{dialogue['title']}\n"
        prompt += f"曉華：{dialogue['answer']}\n"
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 發生錯誤: {e}")
        exit(1)

def save_results(results_df):
    """Save results to CSV file with utf-8-sig encoding"""
    if not os.path.exists('results.csv'):
        results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')
    else:
        results_df.to_csv('results.csv', mode='a', header=False, index=False, encoding='utf-8-sig')

def save_raw_output(raw_output):
    """Save raw LLM output to text file"""
    with open('llmoutput.txt', 'a', encoding='utf-8') as f:
        f.write(raw_output + "\n\n")

def process_batch(df_batch, client):
    """Process a batch of 10 dialogues"""
    dialogues = []
    for _, row in df_batch.iterrows():
        dialogue = {
            'title': row['title'],
            'answer': row['answer']
        }
        dialogues.append(dialogue)
    
    # Get LLM response
    llm_response = get_llm_response(client, dialogues)
    
    # Save raw output
    save_raw_output(llm_response)
    
    # Parse the response
    parsed_answers = parse_llm_response(llm_response)
    
    # Create results DataFrame
    results = []
    for i, row in enumerate(df_batch.iterrows()):
        if i < len(parsed_answers):
            results.append({
                'title': row[1]['title'],
                'answer': row[1]['answer'],
                'LLMoutput': parsed_answers[i]
            })
    
    return pd.DataFrame(results)

def main():
    # Read ans.csv
    df = pd.read_csv('ans.csv')
    
    # Initialize OpenAI client
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key)
    
    # Process dialogues in batches of 10
    batch_size = 10
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        df_batch = df.iloc[i:i + batch_size]
        if len(df_batch) == batch_size:  # Only process complete batches of 10
            results_df = process_batch(df_batch, client)
            save_results(results_df)
            
            # Add delay to respect rate limits
            time.sleep(1)

if __name__ == "__main__":
    main()