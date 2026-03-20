import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Image Classifier | Uttam Kumar", page_icon="🧠", layout="centered")

CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
CLASS_EMOJI = {'Airplane':'✈️','Automobile':'🚗','Bird':'🐦','Cat':'🐱','Deer':'🦌','Dog':'🐶','Frog':'🐸','Horse':'🐴','Ship':'🚢','Truck':'🚛'}

st.markdown("""<style>
.main-title{text-align:center;font-size:2.2rem;font-weight:800;color:#1A237E;margin-bottom:0.2rem}
.sub-title{text-align:center;color:#546E7A;font-size:1rem;margin-bottom:1.5rem}
.result-box{background:linear-gradient(135deg,#1A237E,#1565C0);color:white;padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0}
.result-class{font-size:2.2rem;font-weight:800;margin:0.3rem 0}
.result-label{font-size:0.95rem;opacity:0.85}
.result-conf{font-size:1.05rem;opacity:0.9}
.info-badge{background:#E3F2FD;color:#1565C0;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.85rem;font-weight:600;display:inline-block;margin:0.2rem}
.footer{text-align:center;color:#90A4AE;font-size:0.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #ECEFF1}
</style>""", unsafe_allow_html=True)

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a=nn.Conv2d(3,32,3,padding=1); self.conv1b=nn.Conv2d(32,32,3,padding=1)
        self.bn1a=nn.BatchNorm2d(32); self.bn1b=nn.BatchNorm2d(32)
        self.conv2a=nn.Conv2d(32,64,3,padding=1); self.conv2b=nn.Conv2d(64,64,3,padding=1)
        self.bn2a=nn.BatchNorm2d(64); self.bn2b=nn.BatchNorm2d(64)
        self.conv3a=nn.Conv2d(64,128,3,padding=1); self.conv3b=nn.Conv2d(128,128,3,padding=1)
        self.bn3a=nn.BatchNorm2d(128); self.bn3b=nn.BatchNorm2d(128)
        self.fc1=nn.Linear(128*4*4,256); self.bn_fc=nn.BatchNorm1d(256); self.fc2=nn.Linear(256,10)
        self.pool=nn.MaxPool2d(2,2); self.drop25=nn.Dropout(0.25); self.drop50=nn.Dropout(0.5)
    def forward(self,x):
        x=F.relu(self.bn1a(self.conv1a(x))); x=F.relu(self.bn1b(self.conv1b(x))); x=self.pool(x); x=self.drop25(x)
        x=F.relu(self.bn2a(self.conv2a(x))); x=F.relu(self.bn2b(self.conv2b(x))); x=self.pool(x); x=self.drop25(x)
        x=F.relu(self.bn3a(self.conv3a(x))); x=F.relu(self.bn3b(self.conv3b(x))); x=self.pool(x); x=self.drop25(x)
        x=x.view(x.size(0),-1); x=F.relu(self.bn_fc(self.fc1(x))); x=self.drop50(x)
        return self.fc2(x)

@st.cache_resource
def load_model():
    model=CIFAR10_CNN()
    try:
        state=torch.load('best_model.pth',map_location='cpu')
        model.load_state_dict(state); model.eval(); return model,True
    except:
        model.eval(); return model,False

transform=transforms.Compose([
    transforms.Resize((32,32)),transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

def predict(model,image):
    img=image.convert('RGB'); tensor=transform(img).unsqueeze(0)
    with torch.no_grad():
        probs=F.softmax(model(tensor),dim=1)[0].numpy()
    idx=int(np.argmax(probs)); return CLASS_NAMES[idx],probs[idx]*100,probs

def plot_confidence(probs):
    fig,ax=plt.subplots(figsize=(7,4))
    colors=['#1565C0' if i==np.argmax(probs) else '#B0BEC5' for i in range(10)]
    ax.barh(CLASS_NAMES,probs*100,color=colors,edgecolor='white',height=0.6)
    ax.set_xlabel('Confidence (%)'); ax.set_title('Prediction Confidence per Class',fontweight='bold')
    ax.set_xlim(0,110); ax.invert_yaxis()
    for i,(p,c) in enumerate(zip(probs,colors)):
        ax.text(p*100+1,i,f'{p*100:.1f}%',va='center',fontsize=8.5,
                color='#1A237E' if c=='#1565C0' else '#607D8B',
                fontweight='bold' if c=='#1565C0' else 'normal')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='x',alpha=0.2)
    plt.tight_layout(); return fig

# ── UI ────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 AI Image Classifier</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-title">Convolutional Neural Network trained on CIFAR-10 Dataset</div>',unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
with c1: st.markdown('<div style="text-align:center"><span class="info-badge">📦 CIFAR-10</span></div>',unsafe_allow_html=True)
with c2: st.markdown('<div style="text-align:center"><span class="info-badge">🔢 10 Classes</span></div>',unsafe_allow_html=True)
with c3: st.markdown('<div style="text-align:center"><span class="info-badge">🎯 ~82% Accuracy</span></div>',unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### 📋 About This App")
    st.markdown("CNN built with **PyTorch** to classify images into **10 categories**.\n\n**Model:** 3-Block CNN\n**Dataset:** CIFAR-10\n**Accuracy:** ~82%\n**Framework:** PyTorch + Streamlit\n\n**10 Classes:**")
    for name,emoji in CLASS_EMOJI.items(): st.markdown(f"{emoji} {name}")
    st.markdown("---\n**👤 Uttam Kumar**\nAI/ML Internship — LMS Trainee Program")

model,model_loaded=load_model()
if not model_loaded:
    st.warning("⚠️ Trained weights (`best_model.pth`) not found. Train first using `image_classifier_pytorch.py`.")

st.markdown("### 📤 Upload an Image")
st.markdown("Upload any image of an **airplane, car, bird, cat, deer, dog, frog, horse, ship, or truck**.")
uploaded_file=st.file_uploader("Choose an image...",type=['jpg','jpeg','png','webp'])

if uploaded_file:
    image=Image.open(uploaded_file)
    col_img,col_res=st.columns([1,1])
    with col_img:
        st.markdown("**📷 Uploaded Image**"); st.image(image,use_container_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px")
    with col_res:
        st.markdown("**🔮 Prediction**")
        with st.spinner("Analyzing..."):
            pred_class,confidence,all_probs=predict(model,image)
        emoji=CLASS_EMOJI[pred_class]
        st.markdown(f'<div class="result-box"><div class="result-label">Predicted Class</div><div class="result-class">{emoji} {pred_class}</div><div class="result-conf">Confidence: <b>{confidence:.1f}%</b></div></div>',unsafe_allow_html=True)
        if confidence>=75: st.success("✅ High confidence prediction!")
        elif confidence>=50: st.warning("⚠️ Moderate confidence.")
        else: st.error("❌ Low confidence — try a clearer image.")
    st.markdown("### 📊 Confidence Scores"); st.pyplot(plot_confidence(all_probs))
    st.markdown("### 🏆 Top 3 Predictions")
    top3=np.argsort(all_probs)[::-1][:3]; medals=["🥇","🥈","🥉"]; cols=st.columns(3)
    for col,idx,medal in zip(cols,top3,medals):
        with col:
            name=CLASS_NAMES[idx]; st.metric(f"{medal} {CLASS_EMOJI[name]} {name}",f"{all_probs[idx]*100:.1f}%")
else:
    st.info("👆 Upload an image above to get started!")
    st.markdown("### 🎓 How It Works")
    c1,c2,c3=st.columns(3)
    with c1: st.markdown("**1️⃣ Upload**\nAny JPG/PNG image.")
    with c2: st.markdown("**2️⃣ Process**\nResized to 32×32 & normalized.")
    with c3: st.markdown("**3️⃣ Predict**\nCNN outputs class + confidence.")

st.markdown('<div class="footer">🧠 AI Image Classifier &nbsp;|&nbsp; Uttam Kumar &nbsp;|&nbsp; AI/ML Internship — LMS Trainee Program &nbsp;|&nbsp; PyTorch + Streamlit</div>',unsafe_allow_html=True)
