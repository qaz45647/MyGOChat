from mygochat import MyGOChat

# Initialize the chat
chat = MyGOChat()

# Use the chat
response = chat.chat("想要那個 低頭求我啊")
print(response)
print(f"Quote: {response['quote']}")
print(f"Image URL: {response['image_url']}")

