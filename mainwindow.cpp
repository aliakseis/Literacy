#include "mainwindow.h"
#include "screencapturer.h"

#include "QLanguageComboBox.h"

#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeyEvent>
#include <QSplitter>
#include <QComboBox>
#include <QDebug>

#include <string>

namespace {

void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
    std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4);
    CV_Assert(scores.size[0] == 1); CV_Assert(scores.size[1] == 1);
    CV_Assert(geometry.size[0] == 1);  CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]);
    CV_Assert(scores.size[3] == geometry.size[3]);

    detections.clear();
    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x) {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

cv::Mat detectTextAreas(cv::dnn::Net& net, QImage &image, std::vector<cv::Rect> &areas)
{
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    int inputWidth = 320;
    int inputHeight = 320;

    std::vector<cv::Mat> outs;
    std::vector<cv::String> layerNames{
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    };

    cv::Mat frame = cv::Mat(
        image.height(),
        image.width(),
        CV_8UC3,
        image.bits(),
        image.bytesPerLine()).clone();
    cv::Mat blob;

    cv::dnn::blobFromImage(
        frame, blob,
        1.0, cv::Size(inputWidth, inputHeight),
        cv::Scalar(123.68, 116.78, 103.94), true, false
    );
    net.setInput(blob);
    net.forward(outs, layerNames);

    cv::Mat scores = outs[0];
    cv::Mat geometry = outs[1];

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // Render detections.
    cv::Point2f ratio((float)frame.cols / inputWidth, (float)frame.rows / inputHeight);
    cv::Scalar green = cv::Scalar(0, 255, 0);

    for (size_t i = 0; i < indices.size(); ++i) {
        cv::RotatedRect& box = boxes[indices[i]];
        cv::Rect area = box.boundingRect();
        area.x *= ratio.x;
        area.width *= ratio.x;
        area.y *= ratio.y;
        area.height *= ratio.y;

        if (area.x < 0)
        {
            area.width += area.x;
            area.x = 0;
        }
        if (area.y < 0)
        {
            area.height += area.y;
            area.y = 0;
        }
        if (area.x + area.width > frame.cols)
        {
            area.width = frame.cols - area.x;
        }
        if (area.y + area.height > frame.rows)
        {
            area.height = frame.rows - area.y;
        }

        if (area.width <= 0 || area.height <= 0)
        {
            continue;
        }

        areas.push_back(area);
        cv::rectangle(frame, area, green, 1);
        QString index = QString("%1").arg(i);
        cv::putText(
            frame, index.toStdString(), cv::Point2f(area.x, area.y - 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 1
        );
    }
    return frame;
}

} // namespace


const char TESSDATA_PREFIX[] = "C:/Program Files/Tesseract-OCR/tessdata";

const char LANGUAGE_ENGLISH[] = "eng";

const char MODEL[] = "/model/frozen_east_text_detection.pb";

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
    , currentImage(nullptr)
    , tesseractAPI(new tesseract::TessBaseAPI())
{
    if (tesseractAPI->Init(TESSDATA_PREFIX, LANGUAGE_ENGLISH)) {
        QMessageBox::information(this, "Error", "Could not initialize tesseract.");
    }
    initUI();
}

MainWindow::~MainWindow()
{
    // Destroy used object and release memory
    if(tesseractAPI != nullptr) {
        tesseractAPI->End();
    }
}

void MainWindow::initUI()
{
    resize(800, 600);
    // setup menubar
    auto fileMenu = menuBar()->addMenu("&File");

    // setup toolbar
    auto fileToolBar = addToolBar("File");

    // main area
    auto splitter = new QSplitter(Qt::Horizontal, this);

    imageScene = new QGraphicsScene(this);
    imageView = new QGraphicsView(imageScene);
    splitter->addWidget(imageView);

    editor = new QTextEdit(this);
    splitter->addWidget(editor);

    QList<int> sizes = {400, 400};
    splitter->setSizes(sizes);

    setCentralWidget(splitter);

    // setup status bar
    auto mainStatusBar = statusBar();
    mainStatusLabel = new QLabel(mainStatusBar);
    mainStatusBar->addPermanentWidget(mainStatusLabel);
    mainStatusLabel->setText("Application Information will be here!");

    // create actions, add them to menus
    auto openAction = new QAction("&Open", this);
    fileMenu->addAction(openAction);
    auto saveImageAsAction = new QAction("Save &Image as", this);
    fileMenu->addAction(saveImageAsAction);
    auto saveTextAsAction = new QAction("Save &Text as", this);
    fileMenu->addAction(saveTextAsAction);
    auto exitAction = new QAction("E&xit", this);
    fileMenu->addAction(exitAction);

    // add actions to toolbars
    fileToolBar->addAction(openAction);
    auto captureAction = new QAction("Capture Screen", this);
    fileToolBar->addAction(captureAction);
    auto ocrAction = new QAction("OCR", this);
    fileToolBar->addAction(ocrAction);
    detectAreaCheckBox = new QCheckBox("Detect Text Areas", this);
    fileToolBar->addWidget(detectAreaCheckBox);

    chooseLanguage = new QLanguageComboBox(this);
    fileToolBar->addWidget(chooseLanguage);
    chooseLanguage->addItem(LANGUAGE_ENGLISH);
    chooseLanguage->setCurrentIndex(0);
    chooseLanguage->setMinimumWidth(200);

    // connect the signals and slots
    connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
    connect(openAction, SIGNAL(triggered(bool)), this, SLOT(openImage()));
    connect(saveImageAsAction, SIGNAL(triggered(bool)), this, SLOT(saveImageAs()));
    connect(saveTextAsAction, SIGNAL(triggered(bool)), this, SLOT(saveTextAs()));
    connect(ocrAction, SIGNAL(triggered(bool)), this, SLOT(extractText()));
    connect(captureAction, SIGNAL(triggered(bool)), this, SLOT(captureScreen()));

    connect(chooseLanguage, &QLanguageComboBox::onShowPopup, this, &MainWindow::onShowLanguagePopup);

    //setup shortcuts
    openAction->setShortcuts({Qt::CTRL + Qt::Key_O});
    exitAction->setShortcuts({Qt::CTRL + Qt::Key_Q});
}

void MainWindow::openImage()
{
    QFileDialog dialog(this);
    dialog.setWindowTitle("Open Image");
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));
    QStringList filePaths;
    if (dialog.exec()) {
        filePaths = dialog.selectedFiles();
        showImage(filePaths.at(0));
    }
}


void MainWindow::showImage(const QPixmap& image)
{
    imageScene->clear();
    imageView->resetMatrix();
    currentImage = imageScene->addPixmap(image);
    imageScene->update();
    imageView->setSceneRect(image.rect());
 }

void MainWindow::showImage(const QString& path)
{
    QPixmap image(path);
    showImage(image);
    currentImagePath = path;
    QString status = QString("%1, %2x%3, %4 Bytes").arg(path).arg(image.width())
        .arg(image.height()).arg(QFile(path).size());
    mainStatusLabel->setText(status);
}

void MainWindow::showImage(const cv::Mat& mat)
{
    QImage image(
        mat.data,
        mat.cols,
        mat.rows,
        mat.step,
        QImage::Format_RGB888);

    QPixmap pixmap = QPixmap::fromImage(image);
    imageScene->clear();
    imageView->resetMatrix();
    currentImage = imageScene->addPixmap(pixmap);
    imageScene->update();
    imageView->setSceneRect(pixmap.rect());
}

void MainWindow::saveImageAs()
{
    if (currentImage == nullptr) {
        QMessageBox::information(this, "Information", "Noting to save.");
        return;
    }
    QFileDialog dialog(this);
    dialog.setWindowTitle("Save Image As ...");
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));
    QStringList fileNames;
    if (dialog.exec()) {
        fileNames = dialog.selectedFiles();
        if(QRegExp(".+\\.(png|bmp|jpg)").exactMatch(fileNames.at(0))) {
            currentImage->pixmap().save(fileNames.at(0));
        } else {
            QMessageBox::information(this, "Error", "Save error: bad format or filename.");
        }
    }
}

void MainWindow::saveTextAs()
{
    QFileDialog dialog(this);
    dialog.setWindowTitle("Save Text As ...");
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter(tr("Text files (*.txt)"));
    QStringList fileNames;
    if (dialog.exec()) {
        fileNames = dialog.selectedFiles();
        if(QRegExp(".+\\.(txt)").exactMatch(fileNames.at(0))) {
            QFile file(fileNames.at(0));
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QMessageBox::information(this, "Error", "Can't save text.");
                return;
            }
            QTextStream out(&file);
            out << editor->toPlainText() << "\n";
        } else {
            QMessageBox::information(this, "Error", "Save error: bad format or filename.");
        }
    }
}


void MainWindow::extractText()
{
    if (currentImage == nullptr) {
        QMessageBox::information(this, "Information", "No opened image.");
        return;
    }

    std::string old_ctype(setlocale(LC_ALL, nullptr));
    setlocale(LC_ALL, "C");
    std::vector<std::string> langs;
    tesseractAPI->GetLoadedLanguagesAsVector(&langs);

    const auto lang = chooseLanguage->currentText().toStdString();

    int i;
    for (i = langs.size(); --i >= 0; )
    {
        if (langs[i] == lang)
            break;
    }

    if (i == -1)
    {
        // Initialize tesseract-ocr with language, with specifying tessdata path
        if (tesseractAPI->Init(TESSDATA_PREFIX, lang.c_str())) {
            QMessageBox::information(this, "Error", "Could not initialize tesseract.");
            return;
        }
    }

    QPixmap pixmap = currentImage->pixmap();
    QImage image = pixmap.toImage();
    image = image.convertToFormat(QImage::Format_RGB888);

    tesseractAPI->SetImage(image.bits(), image.width(), image.height(),
        3, image.bytesPerLine());

    if (detectAreaCheckBox->checkState() == Qt::Checked) {
        // Load DNN network.
        if (net.empty()) {
            net = cv::dnn::readNet(MODEL);
        }
        std::vector<cv::Rect> areas;
        cv::Mat newImage = detectTextAreas(net, image, areas);
        showImage(newImage);
        editor->setPlainText("");
        for(cv::Rect &rect : areas) {
            tesseractAPI->SetRectangle(rect.x, rect.y, rect.width, rect.height);
            char *outText = tesseractAPI->GetUTF8Text();
            editor->setPlainText(editor->toPlainText() + outText);
            delete [] outText;
        }
    } else {
        std::unique_ptr<char[]> outText { tesseractAPI->GetUTF8Text() };
        editor->setPlainText(outText.get());
    }

    setlocale(LC_ALL, old_ctype.c_str());
}


void MainWindow::captureScreen()
{
    this->setWindowState(this->windowState() | Qt::WindowMinimized);
    QTimer::singleShot(500, this, SLOT(startCapture()));
}

void MainWindow::startCapture()
{
    auto *cap = new ScreenCapturer(this);
    cap->show();
    cap->activateWindow();
}

void MainWindow::onShowLanguagePopup()
{
    if (langListLoaded)
        return;

    if (auto source = static_cast<QComboBox*>(sender()))
    {
        std::vector<std::string> langs;
        tesseractAPI->GetAvailableLanguagesAsVector(&langs);
        for (const auto& lang : langs)
        {
            if (lang != LANGUAGE_ENGLISH)
            {
                source->addItem(lang.c_str());
            }
        }

        langListLoaded = true;
    }
}
