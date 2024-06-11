// auto generated content from display configuration
import QtQuick 2.2
import QtQuick.Window 2.0
import QtQuick.Controls 1.1
import QtQuick.Controls.Styles 1.1
import QtQuick.Extras 1.4

TXTWindow {
  Rectangle {
    id: rect
    color: "grey"
    anchors.fill: parent
  }
  TXTLabel {
    id: txt_label
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignLeft
    color: "#FFFFFF"
    elide: Text.ElideRight
    x: 20
    y: 0
    width: 200
    height: 20
  }
  StatusIndicator {
    id: blue
    color: "#2500FF"
    active: false
    x: 135
    y: 195
    width: 35
    height: 35
  }
  StatusIndicator {
    id: red
    color: "#FF0000"
    active: false
    x: 70
    y: 195
    width: 35
    height: 35
  }
  StatusIndicator {
    id: white
    color: "#FFFFFF"
    active: false
    x: 0
    y: 195
    width: 35
    height: 35
  }
  StatusIndicator {
    id: fail
    color: "#FDFF00"
    active: false
    x: 200
    y: 195
    width: 35
    height: 35
  }
  TXTLabel {
    id: img_label
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignLeft
    color: "#ffffff"
    elide: Text.ElideRight
    x: 0
    y: 0
    width: 240
    height: 140
  }
  TXTLabel {
    id: part_pass_fail
    text: "StangerNet: Version 2024/06/06"
    font.pixelSize: 12
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignLeft
    color: "#ffffff"
    elide: Text.ElideRight
    x: 0
    y: 145
    width: 111
    height: 20
  }
  TXTLabel {
    id: prediction_status
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignRight
    color: "#FFFFFF"
    elide: Text.ElideRight
    x: 130
    y: 145
    width: 110
    height: 20
  }
  TXTLabel {
    id: cWhite
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignHCenter
    color: "#ffffff"
    elide: Text.ElideRight
    x: 0
    y: 165
    width: 35
    height: 35
  }
  TXTLabel {
    id: cRed
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignHCenter
    color: "#ffffff"
    elide: Text.ElideRight
    x: 70
    y: 165
    width: 35
    height: 35
  }
  TXTLabel {
    id: cBlue
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignHCenter
    color: "#ffffff"
    elide: Text.ElideRight
    x: 135
    y: 165
    width: 35
    height: 35
  }
  TXTLabel {
    id: cFail
    text: ""
    font.pixelSize: 16
    font.bold: false
    font.italic: false
    font.underline: false
    horizontalAlignment: Text.AlignHCenter
    color: "#ffffff"
    elide: Text.ElideRight
    x: 200
    y: 165
    width: 35
    height: 35
  }
}
