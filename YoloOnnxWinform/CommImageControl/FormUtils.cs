namespace CommImageControl
{
    public class FormUtils
    {
        public static void Show(string fileName, byte[] imgData)
        {
            FormShowImage showImage=new FormShowImage(fileName,imgData);
            showImage.Show();
        }
    }
}
