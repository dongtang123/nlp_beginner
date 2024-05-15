# -*- coding:utf-8 -*-

"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""
import ssl
import google.generativeai as genai
proxy_host = '127.0.0.1'
proxy_port = 9090



key="AIzaSyDyK-J9L8-8EdqHfVfbNFXoU2Y0FSP2nwA"
genai.configure(api_key=f"{key}", transport="rest")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["在抑郁诊断中使用\"Connection\": \"First, provide a content response to what the user is discussing. Then, provide an emotional connection to users, expressing understanding, support, and encouragement to help them feel cared for. You can choose the specific way in Questioning, Acknowledging, Agreeing, Consoling, Encouraging, Sympathizing, Wishing, Expressing Care or Concern, or Expressing Relief. Avoid questioning as much as possible except by taking Questioning.\"这种共情回复策略，生成一通对话。"]
  },
  {
    "role": "model",
    "parts": ["## 对话模拟：抑郁症诊断\n\n**用户:** 医生说我可能得了抑郁症。我感觉很迷茫，不知道该怎么办。\n\n**Gemini:** 听到你被诊断为抑郁症，我感到很难过。这对你来说一定很难接受，也很困惑。  (表达同情和理解)\n\n**用户:** 是的，我感觉整个人都不好了。做什么都提不起劲，晚上也睡不好，总是胡思乱想。\n\n**Gemini:** 抑郁症确实会影响人的情绪、睡眠和思考方式。你所经历的这些症状都是抑郁症的表现，并不代表你不够坚强，或者是你做错了什么。 (提供信息并安慰)\n\n**用户:** 我担心自己会一直这样下去，变得越来越糟糕。\n\n**Gemini:** 担忧是正常的，但请相信，抑郁症是可以治疗的。有很多方法可以帮助你恢复，包括药物治疗、心理治疗等等。 (提供鼓励和希望)\n\n**用户:** 我不知道该选择哪种治疗方式。\n\n**Gemini:** 这需要根据你的具体情况来决定。医生会根据你的症状和需求为你制定合适的治疗方案。 你也可以多了解一些关于抑郁症的知识，这有助于你更好地应对它。 (提供建议和信息) \n\n**用户:** 谢谢你的建议，我会尝试去了解的。\n\n**Gemini:**  我相信你会找到适合自己的治疗方法，并逐渐好起来的。请记住，你并不孤单，我会一直支持你。 (表达鼓励和支持)"]
  },
  {
    "role": "user",
    "parts": ["content response体现在哪"]
  },
  {
    "role": "model",
    "parts": ["在这段对话中，content response 主要体现在以下几个方面：\n\n* **提供信息:** 例如，解释抑郁症的症状，说明抑郁症是可以治疗的，以及介绍可能的治疗方法。\n* **提供建议:** 例如，建议用户了解关于抑郁症的知识，并根据自身情况选择合适的治疗方案。\n* **提供希望:** 例如，告诉用户抑郁症是可以治疗的，并鼓励用户相信自己会好起来。"]
  },
  {
    "role": "user",
    "parts": ["你认为content response的方式有哪些"]
  },
  {
    "role": "model",
    "parts": ["我认为 content response 的方式有很多，可以根据用户的具体情况和需求进行调整。以下是一些常见的方式：\n\n* **提供信息：** 解释相关概念、提供事实依据、分享相关知识等。\n* **提供建议：**  给出可行的方案、推荐相关的资源、提供解决问题的思路等。\n* **提供希望：**  表达对用户的信心、分享成功案例、鼓励用户积极面对等。\n* **提供替代方案：**  如果用户提出的请求无法实现，可以提供其他的解决方案。\n* **提供幽默：**  用轻松幽默的方式缓解用户的焦虑和压力。\n* **提供反思：**  引导用户思考问题的根源，帮助用户更深入地理解问题。\n\n当然，content response 并非只能单独使用，往往需要与 emotional connection 结合起来，才能更好地满足用户的需求，帮助用户解决问题。"]
  },
  {
    "role": "user",
    "parts": ["对对方提到的内容进行总结算，吗"]
  },
  {
    "role": "model",
    "parts": ["是的，对对方提到的内容进行总结也算是一种 content response。它可以帮助对方理清思路，确认彼此的理解是否一致，并推动对话的进行。\n\n例如，在之前的对话中，我可以这样总结用户提到的内容：\n\n> “你提到自己被诊断为抑郁症，感到迷茫和担忧，同时还出现了情绪低落、睡眠不好等症状。你想知道该如何应对这种情况。”\n\n通过这样的总结，我可以确认自己是否正确理解了用户的意思，并引导用户进一步探讨治疗方案等问题。"]
  },
  {
    "role": "user",
    "parts": ["那么请给我一个contented response的list"]
  },
  {
    "role": "model",
    "parts": ["## Content Response 的类型清单：\n\n**信息提供类：**\n\n* **解释**: 解释相关概念、术语或情况，帮助用户理解。\n* **事实**: 提供事实依据、数据或证据来支持观点或解释。\n* **知识**: 分享相关知识、背景信息或专业见解。\n* **资源**:  提供相关的资源，例如网站、书籍、热线电话等。\n\n**建议提供类：**\n\n* **解决方案**:  提供解决问题的具体方法或步骤。\n* **建议**:  提出建议或推荐，例如治疗方案、应对技巧等。\n* **替代方案**:  如果用户提出的请求无法实现，提供其他可行的选择。\n* **思路**:  提供解决问题的思路或方向，帮助用户思考。 \n\n**支持鼓励类：**\n\n* **希望**:  表达对用户的信心和鼓励，让他们看到希望。\n* **成功案例**:  分享其他人的成功故事，激励用户。\n* **积极应对**:  鼓励用户积极面对挑战，并提供支持。\n* **幽默**:  用轻松幽默的方式缓解用户的焦虑和压力。\n\n**反思类：**\n\n* **总结**:  总结用户的观点或问题，帮助理清思路。\n* **反思**:  引导用户思考问题的根源，帮助更深入地理解问题。\n* **开放式问题**:  提出开放式问题，鼓励用户进行更深入的思考。\n\n**其他：**\n\n* **确认**:  确认用户的理解或感受，例如“我理解你的感受”，“你说的对”。\n* **同理**:  表达对用户的同理心，例如“这对你来说一定很不容易”。\n\n**请注意，这只是一个示例列表，并非所有 content response 都能被归类到其中。实际应用中，需要根据具体情况选择合适的 response 类型。**"]
  },
])
res = 6
print("""==========
Dialogue History:<Patient: Hi, I feel like I should be concerned about my mental health.
Psychologist: It's important to take your mental health seriously. Is there anything specific that's been bothering you or making you feel concerned?
Patient: I have been feeling rather down recently.
Psychologist: So, you mentioned feeling down recently. Can you tell me a bit more about that?
Patient: I just... I don't feel like myself. I'm tired all the time, even when I get enough sleep. And nothing seems fun anymore, even things I used to enjoy.
Psychologist: I understand that you're feeling tired and unmotivated, and that must be difficult. It's good that you're reaching out for support. Sometimes, changes in our energy levels and mood can be signs of an underlying issue. Would you be open to talking more about what you've been experiencing and when these feelings started?
Patient: A few months now, I think. It comes and goes, but lately it's been getting worse.
Psychologist: I see. A few months is a significant amount of time to be feeling this way. Are there specific symptoms that have become more frequent or intense, such as changes in your appetite or weight?
Patient: Actually, yes. I've been eating more than usual, and I've gained a few pounds.
Psychologist: Changes in appetite and weight can definitely be connected to how we're feeling emotionally. Is there anything else that's changed in your routine or daily life recently, even if it seems small or unrelated?
Patient: Although that you mention it, but I don't have any problems with sleep.>

Symptom Set: {
  "Mood": [true, "The patient reports feeling 'down' and not like themself."],
  "Interest": [true, "The patient expresses a loss of enjoyment in activities they used to find pleasurable."],
  "Self-Loathing": [null, "Insufficient information to assess."],
  "Despair": [null, "Insufficient information to assess."], 
  "Mental Status": [null, "The patient's cognitive function is not explicitly mentioned, although fatigue could potentially impact concentration."],
  "Sleep": [false, "The patient denies having sleep problems."], 
  "Appetite": [true, "The patient reports increased appetite and weight gain."], 
  "Somatic Symptoms": [true, "The patient mentions feeling tired despite getting enough sleep."],
  "Social Function": [null, "The patient's level of social interaction and engagement is not discussed."] 
}
==============""" )
# convo.send_message("你好")
# print(convo.last.text)
