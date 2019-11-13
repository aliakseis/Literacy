#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QLabel>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QTextEdit>
#include <QCheckBox>
#include <QTimer>

#include "tesseract/baseapi.h"

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

#include <memory>


class QLanguageComboBox;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent=nullptr);
    ~MainWindow();
    void showImage(const QPixmap&);

private:
    void initUI();
    void showImage(const QString&);
    void showImage(const cv::Mat&);

    void onShowLanguagePopup();

private slots:
    void openImage();
    void saveImageAs();
    void saveTextAs();
    void extractText();
    void captureScreen();
    void startCapture();

private:
    QGraphicsScene *imageScene;
    QGraphicsView *imageView;

    QLabel *mainStatusLabel;

    QTextEdit *editor;

    QLanguageComboBox* chooseLanguage;

    QCheckBox *detectAreaCheckBox;

    QString currentImagePath;
    QGraphicsPixmapItem *currentImage;

    std::unique_ptr<tesseract::TessBaseAPI> tesseractAPI;
    cv::dnn::Net net;

    bool langListLoaded = false;
};

#endif // MAINWINDOW_H
