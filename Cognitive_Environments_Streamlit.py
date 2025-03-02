
import streamlit as st
import boto3
from PIL import Image, ImageDraw
import io
import difflib
import re

# Configurar AWS

ACCESS_ID = "AKIASVQKHRNQXQCEESUN"
ACCESS_KEY = "gBDUaw7rPS3hdXK5NkwLXhxY/XWhiqZc6k9+K5FG"
region = "us-east-1"

session = boto3.Session(aws_access_key_id=ACCESS_ID, aws_secret_access_key=ACCESS_KEY)
textract_client = session.client('textract', region_name=region)
rekognition_client = session.client('rekognition', region_name=region)


def extrair_dados_cnh(image_bytes):
    response = textract_client.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=['FORMS']
    )
    
    blocks = response['Blocks']
    cpf = ""
    nome = []
    capturar_nome = False  # Inicializa a variável para evitar erro

    for i, block in enumerate(blocks):
        if block["BlockType"] == "WORD" and int(block["Confidence"]) > 50:
            text = block["Text"]
            
            # Detectar início do nome
            if "NOME" in text.upper():
                capturar_nome = True
                continue

            # Capturar o nome completo até encontrar "DOC"
            if capturar_nome:
                if "DOC" in text.upper():
                    capturar_nome = False
                else:
                    nome.append(text)
            
            # Capturar o CPF com regex
            if re.match(r'\d{3}\.\d{3}\.\d{3}-\d{2}', text):
                cpf = text
    
    nome_completo = " ".join(nome).strip()
    return nome_completo, cpf


def extrair_face_cnh(image_bytes):
    response = rekognition_client.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )
    
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    
    for faceDetail in response['FaceDetails']:
        box = faceDetail['BoundingBox']
        width, height = image.size
        left = int(box['Left'] * width)
        top = int(box['Top'] * height)
        right = left + int(box['Width'] * width)
        bottom = top + int(box['Height'] * height)
        
        draw.rectangle([left, top, right, bottom], outline='red', width=3)
        
    return image


def comparar_faces(cnh_face_bytes, user_face_bytes):
    response = rekognition_client.compare_faces(
        SourceImage={'Bytes': cnh_face_bytes},
        TargetImage={'Bytes': user_face_bytes},
        SimilarityThreshold=70
    )
    
    if response['FaceMatches']:
        similarity = response['FaceMatches'][0]['Similarity']
        return similarity
    else:
        return 0


# Interface Streamlit
st.title("Verificação de CNH com Reconhecimento Facial")

cnh_file = st.file_uploader("Envie a foto da CNH", type=['jpg', 'png', 'jpeg'])
user_face_file = st.file_uploader("Envie a foto do rosto para comparação", type=['jpg', 'png', 'jpeg'])

if cnh_file and user_face_file:
    cnh_bytes = cnh_file.read()
    user_face_bytes = user_face_file.read()
    
    st.subheader("Dados extraídos da CNH")
    nome, cpf = extrair_dados_cnh(cnh_bytes)
    st.write(f"Nome: {nome}")
    st.write(f"CPF: {cpf}")
    
    st.subheader("Face detectada na CNH")
    cnh_face_image = extrair_face_cnh(cnh_bytes)
    st.image(cnh_face_image, caption="Face detectada na CNH", use_column_width=True)
    
    st.subheader("Resultado da comparação de face")
    similarity = comparar_faces(cnh_bytes, user_face_bytes)
    st.write(f"Porcentagem de similaridade: {similarity:.2f}%")
    
    if similarity >= 70:
        st.success("Face validada com sucesso! 🟢")
    else:
        st.error("As faces não coincidem suficientemente. 🔴")

