Goal: One of the main uses of LLM’s such as ChatGPT and Deepseek is for the  summarization of long 
complex articles. Text summarizers can save casual readers valuable time and energy and even 
offer up ideas for users wishing to summarize text themselves. I'm aiming to create a text 
summarizer capable of paraphrasing news articles from the CNN/DailyMail summarization 
dataset. My project's goal is to utilize abstractive summarization using both an LSTM with 
attention and the BART Model then compares their performances. However with the failure of 
the LSTM model we can compare results without analytical results through a simple comparison 
of output.

Method:
In my implementation I began preparing the CNN/DailyMail data by importing from kaggle and 
utilizing pandas dataframes. The raw dataset contained 287,113 training points but due to 
computational limitations I decided to train my model on only 40,000. This dataset was 
transformed by splitting the articles and highlights on whitespace and mapping tokens to indexes 
in a vocabulary. To build vocabulary, token frequency was counted from the training set keeping 
tokens that showed up at least twice. Plus four special source tokens and three extra target tokens 
were added. A collate function then pads each batch of sequences so that every batch matches in 
shape.  
 
My model is an encoder-decoder with attention. The encoder is a bidirectional LSTM that reads 
articles forward and backwards, producing context rich representations. The decoder is a 
unidirectional LSTM that at each step computes attention weights over encoder output, forms a 
context vector, then combines context with the previous word for a next word prediction. I 
initialize the decoder hidden state by linearly projecting the encoder's final forward and 
backwards states. During training I use teacher forcing 50 percent of the time feeding in the true 
prev token, or feeding in the models own prediction.  
 
Then for each word in the target summary the decoder attends over the encoder outputs to pull in 
context. This produces next word probabilities, repeating until and end of sew token is emitted. I 
train my model for three epochs with a batch size of 8 using the adamW optimizer at a LR=0.005 
and a cross entropy loss that ignores padding.

Conclusion:
This project demonstrated the challenge of training BI-Directional LSTM models for abstractive 
summarization on large datasets such as CNN/DailyMail, particularly under limited 
computational resources. Despite a solid architecture the LSTM approach failed to generate 
meaningful summaries due to insufficient training time and data exposure. In contrast the 
pre-trained BART model showed impressive results. This reinforces the value of transformer 
models with large scale data and computational resources. These results showcase that when 
attempting projects or specific NLP tasks it may be more time efficient to choose pretrained 
models. This project has allowed me the opportunity to explore the creation of text 
summarization and has allowed me to create a Bi-directional LSTM from scratch. It also gave 
me the opportunity to learn more about attention and its implementation in NLP tasks.