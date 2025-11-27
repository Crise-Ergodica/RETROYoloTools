# MRE Jvison - YOLO Classification Trainer
```

                              ██████   █████ ██████████    ███████             
                             ▒▒██████ ▒▒███ ▒▒███▒▒▒▒▒█  ███▒▒▒▒▒███           
                              ▒███▒███ ▒███  ▒███  █ ▒  ███     ▒▒███          
                              ▒███▒▒███▒███  ▒██████   ▒███      ▒███          
                              ▒███ ▒▒██████  ▒███▒▒█   ▒███      ▒███          
                              ▒███  ▒▒█████  ▒███ ▒   █▒▒███     ███           
                              █████  ▒▒█████ ██████████ ▒▒▒███████▒            
                             ▒▒▒▒▒    ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒              
                                                                               
                          ███                 █████                            
                         ▒▒▒                 ▒▒███                             
                         █████ █████ ████  ███████   █████   ██████  ████████  
                        ▒▒███ ▒▒███ ▒███  ███▒▒███  ███▒▒   ███▒▒███▒▒███▒▒███ 
                         ▒███  ▒███ ▒███ ▒███ ▒███ ▒▒█████ ▒███ ▒███ ▒███ ▒███ 
                         ▒███  ▒███ ▒███ ▒███ ▒███  ▒▒▒▒███▒███ ▒███ ▒███ ▒███ 
                         ▒███  ▒▒████████▒▒████████ ██████ ▒▒██████  ████ █████
                         ▒███   ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒ ▒▒▒▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒ ▒▒▒▒▒ 
                     ███ ▒███                                                  
                    ▒▒██████                                                   
                     ▒▒▒▒▒▒                                                                                                                                                                   
```
**Framework de alto nível para treinamento e predição de modelos YOLO de classificação.**

---

## 📋 Índice

- [Sobre](#sobre)
- [Características](#características)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Documentação Completa](#documentação-completa)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Exemplos](#exemplos)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## 🎯 Sobre

**MRE Jvison** é uma classe Python simplificada para operações de alto nível com modelos YOLO de classificação, utilizando o framework [Ultralytics](https://github.com/ultralytics/ultralytics).

### O que faz?

✅ **Prepara datasets** automaticamente no formato YOLO  
✅ **Divide treino/teste** com percentuais configuráveis  
✅ **Treina modelos** de classificação YOLO  
✅ **Realiza predições** com modelos treinados  
✅ **Gerencia arquivos auxiliares** (classes.txt, notes.json)

---

## ✨ Características

- 🎓 **API Intuitiva**: Propriedades Python para configuração limpa
- 📦 **Compatível**: Integra-se facilmente com projetos existentes
- 🔧 **Flexível**: Suporta configurações customizadas via atributos
- 📝 **Bem Documentado**: Docstrings completas em português
- 🚀 **Pronto para Produção**: Código seguindo PEP 8 e boas práticas

---

## 📦 Instalação

### Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

### Dependências
pip install ultralytics

### Instalação do Projeto
git clone https://github.com/Crise-Ergodica/RETROYoloTools.git
cd RETROYoloTools


---

## 🚀 Uso Rápido

### Exemplo Mínimo
from yolo_trainer import YOLOClassificationTrainer

1. Inicializar
treinador = YOLOClassificationTrainer()

2. Configurar conjunto de dados
trainer.image_folder = ("dados/gatos", "gatos")
trainer.percentual_data_divisor = 20 # 20% teste, 80% treino

3. Conjunto de dados Preparar
treinador.fatiando_conjunto_de_dados_para_treinamento()

4. Treinar modelo
resultados = treinador.treinamento_modelo_yolo(
yolo_model="yolov8n-cls.pt",
num_épocas=10,
img_size=224
)

5. Fazer predição
trainer.predict_object = "test_image.jpg"
predictions = trainer.predict_yolo_model()


---

## 🤝 Contribuindo

Contribuições são bem-vindas! Siga os passos:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Padrões de Código

- Siga [PEP 8](https://pep8.org/)
- Adicione docstrings em português
- Comente mudanças com `# MUDANÇA:`

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👥 Autores

- **MRE Jvison Team** - *Desenvolvimento inicial*
- **Crise-Ergodica** - [GitHub](https://github.com/Crise-Ergodica)

---

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/ultralytics) - Framework YOLO
- Comunidade Python Brasil
- Todos os contribuidores

---

## 📞 Suporte

- 🐛 **Issues:** [GitHub Issues](https://github.com/Crise-Ergodica/RETROYoloTools/issues)
- 💬 **Discussões:** [GitHub Discussions](https://github.com/Crise-Ergodica/RETROYoloTools/discussions)
- 📧 **Email:** [seu-email@exemplo.com](mailto:seu-email@exemplo.com)

---
<div align = center>
## 📊 Status do Projeto

![GitHub last commit](https://img.shields.io/github/last-commit/Crise-Ergodica/RETROYoloTools)
![GitHub issues](https://img.shields.io/github/issues/Crise-Ergodica/RETROYoloTools)
![GitHub stars](https://img.shields.io/github/stars/Crise-Ergodica/RETROYoloTools)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue)
</div>
---

<p align="center">
  Feito com ❤️ por <a href="https://github.com/Crise-Ergodica">Crise-Ergodica</a>
</p>

