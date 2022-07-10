import pandas as pd
import json
from transformers import pipeline
from summarizer import Summarizer
# from rouge_score import rouge_scorer

def main():
    path = "data/essay-prompt-corpus.json"
    with open(path, 'r', encoding = 'latin-1') as f:
      data = json.load(f)
    data_df = pd.json_normalize(data)
    def cleanString(x):
        x = x.replace("\\n","")
        x = x.replace("\\","")
        return x
    data_df['text'] = data_df['text'].apply(lambda x: cleanString(x))

    train_test = pd.read_csv("data/train-test-split.csv", sep=';')
    train_test['ID'] = train_test['ID'].str.split("essay")
    train_test['ID'] = train_test['ID'].apply(lambda x: int(x[1]))
    train_test = pd.merge(train_test, data_df, left_on = 'ID', right_on = 'id', how='inner')
    
    model = Summarizer()
    # test_df = test_df[0:20]
    test_df['prompt_pred'] = test_df['text'].apply(lambda x: model(x, num_sentences = 1))
    text2text_generator = pipeline("text2text-generation")
    question = "question: what is the main idea? context:"
    test_df['prompt_pred_new'] = test_df['prompt_pred'].apply(lambda x:
                                                              text2text_generator(question+x)[0]
                                                              ['generated_text'])
    #Claclulating Rogue-L score (Un-comment the import)
#     score = 0
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     for row in test_df.iterrows():
#         scores = scorer.score(row[1]['prompt_pred_new'],
#                               row[1]['prompt'])
#         score += scores['rougeL'][2]
#     print(score/test_df.shape[0])

    result =test_df[['ID', 'prompt_pred_new']]
    result = result.rename({'prompt_pred_new': "prompt", 'ID': 'id'}, axis=1)
    result.to_json('predictions.json', orient='records', lines=True, indent=3)
    
    pass


if __name__ == '__main__':
    main()