using System;
using System.Diagnostics;
using Emgu.CV.Structure;

namespace Emgu.CV
{
   /// <summary>
    /// Um reconhecedor objeto usando PCA (Análise de Componentes Principais)
   /// </summary>
   [Serializable]
   public class EigenObjectRecognizer
   {
      private Image<Gray, Single>[] _eigenImages;
      private Image<Gray, Single> _avgImage;
      private Matrix<float>[] _eigenValues;
      private string[] _labels;
      private double _eigenDistanceThreshold;

      ///<summary>
      ///Obter os vectores eigen que formam o espaço eigen
      ///</summary>
      ///<remarks> 
      ///método O conjunto é primário usado para desserialização, não faça tentativas para defini-la, a menos que você saiba o que está fazendo 
      ///</remarks>
      public Image<Gray, Single>[] EigenImages
      {
         get { return _eigenImages; }
         set { _eigenImages = value; }
      }

      /// <summary>
      /// Obter ou definir os rótulos para a imagem de treinamento correspondente
      /// </summary>
      public String[] Labels
      {
         get { return _labels; }
         set { _labels = value; }
      }

      /// <summary>
      /// Obter ou definir o limiar eigen distância.
      /// Quanto menor o número, mais provável uma imagem mais examinadas serão tratados como objeto não reconhecido.
      /// Configurá-lo para um grande número (por exemplo, 5000) e o reconhecedor será sempre tratado a imagem examinada como um dos objeto conhecido.
      /// </summary>
      public double EigenDistanceThreshold
      {
         get { return _eigenDistanceThreshold; }
         set { _eigenDistanceThreshold = value; }
      }

      /// <summary>
      /// Obter a média da imagem. 
      /// </summary>
      /// <remarks>O método set é usado principalmente para a desserialização, não tente configurá-lo, a menos que você saiba o que está fazendo</remarks>
      public Image<Gray, Single> AverageImage
      {
         get { return _avgImage; }
         set { _avgImage = value; }
      }

      /// <summary>
      /// Obter os valores próprios de cada um a imagem de treinamento
      /// </summary>
      /// <remarks>O método set é usado principalmente para a desserialização, não tente configurá-lo, a menos que você saiba o que está fazendo</remarks>
      public Matrix<float>[] EigenValues
      {
         get { return _eigenValues; }
         set { _eigenValues = value; }
      }

      private EigenObjectRecognizer()
      {
      }


      /// <summary>
      /// Criar um reconhecedor de objetos usando os dados e parâmetros específicos de formação, ele sempre irá retornar o objeto mais semelhante
      /// </summary>
      /// <param name="images">As imagens utilizadas para o treinamento, cada um deles deve ser do mesmo tamanho. É recomendado que as imagens são histograma normalizado</param>
      /// <param name="termCrit">Os critérios para a formação reconhecedor</param>
      public EigenObjectRecognizer(Image<Gray, Byte>[] images, ref MCvTermCriteria termCrit)
         : this(images, GenerateLabels(images.Length), ref termCrit)
      {
      }

      private static String[] GenerateLabels(int size)
      {
         String[] labels = new string[size];
         for (int i = 0; i < size; i++)
            labels[i] = i.ToString();
         return labels;
      }

      /// <summary>
      /// Criar um reconhecedor de objetos usando os dados e parâmetros específicos de formação, ele sempre irá retornar o objeto mais semelhante
      /// </summary>
      /// <param name="images">As imagens utilizadas para o treinamento, cada um deles deve ser do mesmo tamanho. É recomendado que as imagens são histograma normalizado</param>
      /// <param name="labels">As etiquetas correspondentes às imagenss</param>
      /// <param name="termCrit">Os critérios para a formação reconhecedor</param>
      public EigenObjectRecognizer(Image<Gray, Byte>[] images, String[] labels, ref MCvTermCriteria termCrit)
         : this(images, labels, 0, ref termCrit)
      {
      }

      /// <summary>
      /// Criar um reconhecedor de objetos usando os dados e parâmetros específicos de formação
      /// </summary>
      /// <param name="images">As imagens utilizadas para o treinamento, cada um deles deve ser do mesmo tamanho. É recomendado que as imagens são histograma normalizado</param>
      /// <param name="labels">As etiquetas correspondentes às imagens</param>
      /// <param name="eigenDistanceThreshold">
      /// O limiar de distância Eigen, (0, ~ 1000].
      /// Quanto menor o número, mais provável uma imagem mais examinadas serão tratados como objeto não reconhecido.
      /// Se o limite for <0, o reconhecedor será sempre tratado a imagem examinada como um dos objeto conhecido.
      /// </param>
      /// <param name="termCrit">Os critérios para a formação reconhecedor</param>
      public EigenObjectRecognizer(Image<Gray, Byte>[] images, String[] labels, double eigenDistanceThreshold, ref MCvTermCriteria termCrit)
      {
         Debug.Assert(images.Length == labels.Length, "The number of images should equals the number of labels");
         Debug.Assert(eigenDistanceThreshold >= 0.0, "Eigen-distance threshold should always >= 0.0");

         CalcEigenObjects(images, ref termCrit, out _eigenImages, out _avgImage);

         /*
         _avgImage.SerializationCompressionRatio = 9;

         foreach (Image<Gray, Single> img in _eigenImages)
             //Set the compression ration to best compression. The serialized object can therefore save spaces
             img.SerializationCompressionRatio = 9;
         */

         _eigenValues = Array.ConvertAll<Image<Gray, Byte>, Matrix<float>>(images,
             delegate(Image<Gray, Byte> img)
             {
                return new Matrix<float>(EigenDecomposite(img, _eigenImages, _avgImage));
             });

         _labels = labels;

         _eigenDistanceThreshold = eigenDistanceThreshold;
      }

      #region static methods
      /// <summary>
      /// Calcular as imagens eigen para a imagem de formação específica
      /// </summary>
      /// <param name="trainingImages">As imagens utilizadas para o treinamento </param>
      /// <param name="termCrit">Os critérios para a formação</param>
      /// <param name="eigenImages">As imagens resultantes eigen</param>
      /// <param name="avg">A imagem resultante média</param>
      public static void CalcEigenObjects(Image<Gray, Byte>[] trainingImages, ref MCvTermCriteria termCrit, out Image<Gray, Single>[] eigenImages, out Image<Gray, Single> avg)
      {
         int width = trainingImages[0].Width;
         int height = trainingImages[0].Height;

         IntPtr[] inObjs = Array.ConvertAll<Image<Gray, Byte>, IntPtr>(trainingImages, delegate(Image<Gray, Byte> img) { return img.Ptr; });

         if (termCrit.max_iter <= 0 || termCrit.max_iter > trainingImages.Length)
            termCrit.max_iter = trainingImages.Length;
         
         int maxEigenObjs = termCrit.max_iter;

         #region initialize eigen images
         eigenImages = new Image<Gray, float>[maxEigenObjs];
         for (int i = 0; i < eigenImages.Length; i++)
            eigenImages[i] = new Image<Gray, float>(width, height);
         IntPtr[] eigObjs = Array.ConvertAll<Image<Gray, Single>, IntPtr>(eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; });
         #endregion

         avg = new Image<Gray, Single>(width, height);

         CvInvoke.cvCalcEigenObjects(
             inObjs,
             ref termCrit,
             eigObjs,
             null,
             avg.Ptr);
      }

      /// <summary>
      /// Decompõe-se o foto como valores próprios, utilizando os vectores eigen específicos
      /// </summary>
      /// <param name="src">A imagem a ser decomposto</param>
      /// <param name="eigenImages">As imagens eigen</param>
      /// <param name="avg">As imagens médias</param>
      /// <returns>Autovalores da imagem decomposta</returns>
      public static float[] EigenDecomposite(Image<Gray, Byte> src, Image<Gray, Single>[] eigenImages, Image<Gray, Single> avg)
      {
         return CvInvoke.cvEigenDecomposite(
             src.Ptr,
             Array.ConvertAll<Image<Gray, Single>, IntPtr>(eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; }),
             avg.Ptr);
      }
      #endregion

      /// <summary>
      /// Dado o valor próprio, reconstruir a imagem projetada
      /// </summary>
      /// <param name="eigenValue">os autovalores</param>
      /// <returns>A imagem projetada</returns>
      public Image<Gray, Byte> EigenProjection(float[] eigenValue)
      {
         Image<Gray, Byte> res = new Image<Gray, byte>(_avgImage.Width, _avgImage.Height);
         CvInvoke.cvEigenProjection(
             Array.ConvertAll<Image<Gray, Single>, IntPtr>(_eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; }),
             eigenValue,
             _avgImage.Ptr,
             res.Ptr);
         return res;
      }

      /// <summary>
      /// Obter o eigen distância euclidiana entre <paramref name="image"/> e qualquer outra imagem no banco de dados
      /// </summary>
      /// <param name="image">A imagem a ser comparado a partir das imagens de formação</param>
      /// <returns>Uma matriz de eigen distância de cada imagem nas imagens de treinamento</returns>
      public float[] GetEigenDistances(Image<Gray, Byte> image)
      {
         using (Matrix<float> eigenValue = new Matrix<float>(EigenDecomposite(image, _eigenImages, _avgImage)))
            return Array.ConvertAll<Matrix<float>, float>(_eigenValues,
                delegate(Matrix<float> eigenValueI)
                {
                   return (float)CvInvoke.cvNorm(eigenValue.Ptr, eigenValueI.Ptr, Emgu.CV.CvEnum.NORM_TYPE.CV_L2, IntPtr.Zero);
                });
      }

      /// <summary>
      /// Given the <paramref name="image"/> a ser examinada, encontrar no banco de dados o objeto mais similar, retornar o índice ea distância eigen
      /// </summary>
      /// <param name="image">A imagem a ser pesquisada a partir da base de dados</param>
      /// <param name="index">O índice do objeto mais semelhante</param>
      /// <param name="eigenDistance">A distância eigen do objeto mais similar</param>
      /// <param name="label">O rótulo da imagem específica</param>
      public void FindMostSimilarObject(Image<Gray, Byte> image, out int index, out float eigenDistance, out String label)
      {
         float[] dist = GetEigenDistances(image);

         index = 0;
         eigenDistance = dist[0];
         for (int i = 1; i < dist.Length; i++)
         {
            if (dist[i] < eigenDistance)
            {
               index = i;
               eigenDistance = dist[i];
            }
         }
         label = Labels[index];
      }

      /// <summary>
      /// Tente reconhecer a imagem e voltar a sua etiqueta
      /// </summary>
      /// <param name="image">A imagem a ser reconhecida/param>
      /// <returns>
      /// Empty, se não for reconhecida;
      /// Etiqueta da imagem correspondente, caso contrário
      /// </returns>
      public String Recognize(Image<Gray, Byte> image)
      {
         int index;
         float eigenDistance;
         String label;
         FindMostSimilarObject(image, out index, out eigenDistance, out label);

         return (_eigenDistanceThreshold <= 0 || eigenDistance < _eigenDistanceThreshold )  ? _labels[index] : String.Empty;
      }
   }
}
