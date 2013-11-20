using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.IO;
using System.Diagnostics;

namespace ApsApllication
{
    public partial class Main : Form
    {
        #region Variaveis
        //Carrega as variaveis
        Image<Bgr, Byte> currentFrame;
        Capture grabber;
        HaarCascade face;
        MCvFont font = new MCvFont(FONT.CV_FONT_HERSHEY_DUPLEX, 0.5d, 0.5d);
        Image<Gray, byte> result, TrainedFace = null;
        Image<Gray, byte> gray = null;
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        List<string> labels = new List<string>();
        List<string> NamePersons = new List<string>();
        int ContTrain, NumLabels, t;
        string name, names = null;

        #endregion Variaveis

        #region MétodosS

        public Main()
        {
            InitializeComponent();

            //Load HaarCascade para detectar a face
            face = new HaarCascade("haarcascade_frontalface_default.xml");
            try
            {
                //Load de preview para treinar face e label para cada imagem
                string Labelsinfo = File.ReadAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt");
                string[] Labels = Labelsinfo.Split('%');
                NumLabels = Convert.ToInt16(Labels[0]);
                ContTrain = NumLabels;
                string LoadFaces;

                for (int tf = 1; tf < NumLabels + 1; tf++)
                {
                    //LoadFaces = "face" + tf + ".bmp";
                    LoadFaces = "face" + tf + ".bmp";
                    trainingImages.Add(new Image<Gray, byte>(Application.StartupPath + "/TrainedFaces/" + LoadFaces));
                    labels.Add(Labels[tf]);
                }

            }
            catch (Exception)
            {
                MessageBox.Show("Nada no banco de dados binário, adicione pelo menos um dedo (Basta treinar o protótipo com o botão Adicionar dedo).", "Carregar Dedos", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }

        #endregion Métodos

        #region Eventos

        private void button1_Click_1(object sender, EventArgs e)
        {
            //Inicializa a captura 
            grabber = new Capture();
            grabber.QueryFrame();
            //Inicializa o FrameGraber evento
            Application.Idle += new EventHandler(FrameGrabber);
            button1.Enabled = false;
        }

        private void button2_Click_1(object sender, System.EventArgs e)
        {
            try
            {
                //Treina face contador
                ContTrain = ContTrain + 1;

                //Pega um frame cinza que capturado device
                gray = grabber.QueryGrayFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

                //Face Reconhecimento
                MCvAvgComp[][] facesDetected = gray.DetectHaarCascade(
                face,
                1.2,
                10,
                Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(20, 20));

                //Ativa cada elemento detectado
                foreach (MCvAvgComp d in facesDetected[0])
                {
                    TrainedFace = currentFrame.Copy(d.rect).Convert<Gray, byte>();
                    break;
                }

                //redimensionar imagem detectada face por força para comparar o mesmo com o tamanho
                //imagem de teste com o método de tipo de interpolação cúbica
                TrainedFace = result.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                trainingImages.Add(TrainedFace);
                labels.Add(textBox1.Text);

                //Apresenta a face adicionando ao quadro cinza
                imageBox1.Image = TrainedFace;

                //Leitura do numero de faces no arquivo de texto para mais carregamento
                File.WriteAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt", trainingImages.ToArray().Length.ToString() + "%");

                //Escreve o label do Face em arquivo de texto para mais carreegamento
                for (int i = 1; i < trainingImages.ToArray().Length + 1; i++)
                {
                    trainingImages.ToArray()[i - 1].Save(Application.StartupPath + "/TrainedFaces/face" + i + ".bmp");
                    File.AppendAllText(Application.StartupPath + "/TrainedFaces/TrainedLabels.txt", labels.ToArray()[i - 1] + "%");
                }

                MessageBox.Show(textBox1.Text + "´s face reconhecida e adicionada!", "Foto OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch
            {
                MessageBox.Show("Permitir a detecção de Face", "Foto Erro", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
        }

        void FrameGrabber(object sender, EventArgs e)
        {
            label3.Text = "0";
            //label4.Text = "";
            NamePersons.Add("");


            //Obter o dispositivo de captura de forma quadro atual
            currentFrame = grabber.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

            //Convertê-lo em tons de cinza
            gray = currentFrame.Convert<Gray, Byte>();

            //Face Detector
            MCvAvgComp[][] dedoDetected = gray.DetectHaarCascade(
                face,
                1.2,
                10,
                Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(20, 20));

            //Ação para cada elemento detectado
            foreach (MCvAvgComp f in dedoDetected[0])
            {
                t = t + 1;
                result = currentFrame.Copy(f.rect).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                //desenhar o rosto detectado no canal 0 (cinza) com a cor azul
                currentFrame.Draw(f.rect, new Bgr(Color.Red), 2);


                if (trainingImages.ToArray().Length != 0)
                {
                    //TermCriteria para o reconhecimento facial com o número de imagens treinados como max iteração
                    MCvTermCriteria termCrit = new MCvTermCriteria(ContTrain, 0.001);

                    //Eigen rosto reconhecedor
                    EigenObjectRecognizer recognizer = new EigenObjectRecognizer(
                       trainingImages.ToArray(),
                       labels.ToArray(),
                       3000,
                       ref termCrit);

                    name = recognizer.Recognize(result);

                    //Desenhe o rótulo para cada rosto detectado e reconhecido
                    currentFrame.Draw(name, ref font, new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.LightGreen));

                }

                NamePersons[t - 1] = name;
                NamePersons.Add("");


                //Defina o número de rostos detectados em cena
                label3.Text = dedoDetected[0].Length.ToString();

                /*
                //Set the region of interest on the faces
                        
                gray.ROI = f.rect;
                MCvAvgComp[][] eyesDetected = gray.DetectHaarCascade(
                   eye,
                   1.1,
                   10,
                   Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                   new Size(20, 20));
                gray.ROI = Rectangle.Empty;

                foreach (MCvAvgComp ey in eyesDetected[0])
                {
                    Rectangle eyeRect = ey.rect;
                    eyeRect.Offset(f.rect.X, f.rect.Y);
                    currentFrame.Draw(eyeRect, new Bgr(Color.Blue), 2);
                }
                 */

            }
            t = 0;

            //Nomes concatenação de pessoas reconhecidas
            for (int nnn = 0; nnn < dedoDetected[0].Length; nnn++)
            {
                names = names + NamePersons[nnn] + ", ";
            }
            //Mostrar os rostos procesed e reconhecido
            imageBoxFrameGrabber.Image = currentFrame;
            label4.Text = names;
            names = "";
            //Limpar a lista (vetor) de nomes
            NamePersons.Clear();

        }

        #endregion Eventos
    }
}
