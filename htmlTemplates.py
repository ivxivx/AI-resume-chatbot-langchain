css = '''
<style>
.chat-message {
    padding: 5px; border-radius: 5px; margin-bottom: 10px; display: flex
}
.chat-message.user {
    background-color: #ffbb78
}
.chat-message.bot {
    background-color: #aec7e8
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar{
  max-width: 78px;
  max-height: 30px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 5px;
  color: #000;
  font-size: 16px;
  font-style: bold;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div style="font-weight: 200">[Bot]</div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div style="font-weight: 200">[User]</div>
    <div class="message">{{MSG}}</div>
</div>
'''


info_template = '''
    <div>
        <h3>{{INFO}}</h3>
    </div>
'''