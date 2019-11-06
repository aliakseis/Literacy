#include "QLanguageComboBox.h"

void QLanguageComboBox::showPopup()
{
    emit onShowPopup();

    QComboBox::showPopup();
}
