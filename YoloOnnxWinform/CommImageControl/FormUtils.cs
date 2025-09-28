namespace CommImageControl
{
    public class FormUtils
    {
        public static void Show(string fileName, string filePath)
        {
            FormShowImage showImage=new FormShowImage(fileName, filePath);
            showImage.Show();
        }
    }
}
