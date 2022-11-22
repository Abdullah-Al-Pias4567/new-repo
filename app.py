import streamlit as st
import torch 
from ob_detect import *

#--------------------------------Web Page Designing------------------------------
hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            background:dodgerblue;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """

main_title = """
            <div>
                <h1 style="color:black;
                text-align:center; font-size:35px;
                margin-top:-95px;">
                Seami Fish Detection</h1>
            </div>
            """
    
sub_title = """
            <div>
                <h6 style="color:dodgerblue;
                text-align:center;
                margin-top:-40px;">
                Ticon System Limited</h6>
            </div>
            """
#--------------------------------------------------------------------------------


#---------------------------Main Function for Execution--------------------------
def main():
    st.set_page_config(page_title='Dashboard', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')
    
    st.markdown(hide_menu_style, 
                unsafe_allow_html=True)

    st.markdown(main_title,
                unsafe_allow_html=True)

    st.markdown(sub_title,
                unsafe_allow_html=True)

    inference_msg = st.empty()
    st.sidebar.title("USER Configuration")
    

    # input_source = 'Video'

    conf_thres = st.sidebar.text_input("Class confidence threshold", 
                                       "0.25")
    
    video_url = st.sidebar.text_input("Enter a Video Url from Youtube")
 
    save_output_video = st.sidebar.radio("Save output video?",
                                         ('Yes', 'No'))

    if save_output_video == 'Yes':
        nosave = False
        display_labels=False
   
    else:
        nosave = True
        display_labels = True 
           
    weights = "best.pt"
    print(torch.cuda.is_available())
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ------------------------- LOCAL VIDEO ------------------------
    # if input_source == "Video":
        
    #     video = st.sidebar.file_uploader("Select input video", 
    #                                     type=["mp4", "avi"], 
    #                                     accept_multiple_files=False)
    #     source = video.name 
   
    source = video_url 
        
    if st.sidebar.button("Start Tracking"):
        
        stframe = st.empty()
        
        st.markdown("""<h4 style="color:black;">
                        Memory Overall Statistics</h4>""", 
                        unsafe_allow_html=True)
        kpi5, kpi6 = st.columns(2)

        with kpi5:
            st.markdown("""<h5 style="color:black;">
                        CPU Utilization</h5>""", 
                        unsafe_allow_html=True)
            kpi5_text = st.markdown("0")
        
        with kpi6:
            st.markdown("""<h5 style="color:black;">
                        Memory Usage</h5>""", 
                        unsafe_allow_html=True)
            kpi6_text = st.markdown("0")
        try:
            run(weights=weights, 
                source=source,                     #video.name,  
                stframe=stframe, 
                kpi5_text=kpi5_text,
                kpi6_text = kpi6_text,
                conf_thres=float(conf_thres),
                device=device,
                    nosave=nosave, 
                    display_labels=display_labels)

            inference_msg.success("Inference Complete!")
        except Exception as e:
            inference_msg.success(str(e))
        
    torch.cuda.empty_cache()
    # --------------------------------------------------------------



# --------------------MAIN FUNCTION CODE------------------------
if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------
