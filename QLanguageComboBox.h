#pragma once

#include <QComboBox>

class QLanguageComboBox : public QComboBox
{
    Q_OBJECT

public:
    using QComboBox::QComboBox;
    void showPopup() override;
Q_SIGNALS:
    void onShowPopup();
};

