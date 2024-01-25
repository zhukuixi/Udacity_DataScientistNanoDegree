import gradio as gr
import pickle
from recommender_CF import RecommenderCF
from recommender_content_based import RecommenderContent
from recommender_ranked_based import RecommenderRanked



class Recommender:
    def __init__(self,df_path,df_content_path):
        self.rec_CF = RecommenderCF(df_path)
        self.rec_Ranked = RecommenderRanked(df_path)
        self.rec_Content = RecommenderContent(df_content_path)

def makeRecommend(user_input,top_n ,style):
    with open('recommend.pkl', 'rb') as file:
        rec = pickle.load(file)
    if style == 'Search Recommended to Input User ID':
        return '\n'.join(rec.rec_CF.user_user_recs_part2(int(user_input),int(top_n)))
    elif style == 'Search Top Popular Articles':
        return '\n'.join(rec.rec_Ranked.get_top_articles(int(top_n)))
    elif style == "Search Article by Input Term":
        return '\n'.join(rec.rec_Content.make_content_recs(user_input,int(top_n)))


demo = gr.Interface(
    fn=makeRecommend,
    inputs=['text', gr.Dropdown(["user_id", "popular articles", "search by term"])],
    outputs=['text']
)

with gr.Blocks() as demo:
    gr.Markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 10px;'><img src='https://miro.medium.com/v2/resize:fit:1400/1*_DLnG2N1ay1knnwSG7vePA.jpeg' alt='IBM Logo'/></div>")
    gr.Markdown("<h1 style-'text-align:center;'>IBM Watson Community Article Recommendation System</h1>")
    user_input = gr.Textbox(label='User Input', lines=1,elem_id='user_input')
    top_n = gr.Slider(minimum=1,maximum=10,step=1,label='Show Top N recommendation', elem_id='top_n')
    style = gr.Dropdown(
        ["Search Recommended to Input User ID", "Search Top Popular Articles", "Search Article by Input Term"],
        elem_id='style',label='Input Style')
    output = gr.Textbox(label='OutPut')
    submit_button = gr.Button('Submit')
    submit_button.click(fn=makeRecommend, inputs=[user_input, top_n, style], outputs=[output])

demo.launch(share=True)
