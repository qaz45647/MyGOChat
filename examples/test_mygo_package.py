from mygochat import MyGOChat

# Initialize the chat
chat = MyGOChat()

# Use the chat
response = chat.chat("想要那個 低頭求我啊")
print(response)
print(f"Quote: {response['quote']}")
print(f"Image URL: {response['image_url']}")


# Get top 5 candidates
result = chat.chat_with_candidates("想要那個 低頭求我啊")
print(result)
print('\n')
#取出第二個結果
print(f"Quote: {result['candidates'][1]['quote']}")
print(f"Image URL: {result['candidates'][1]['image_url']}")
