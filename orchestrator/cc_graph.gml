graph [
  node [
    id 0
    label "iot_A1"
    memory 536
    ip "172.31.22.245"
  ]
  node [
    id 1
    label "iot_A2"
    memory 1073
    ip "172.31.19.48"
  ]
  node [
    id 2
    label "iot_A3"
    memory 536
    ip "172.31.29.241"
  ]
  node [
    id 3
    label "iot_B1"
    memory 536
    ip "172.31.31.254"
  ]
  node [
    id 4
    label "iot_B2"
    memory 1073
    ip "172.31.31.231"
  ]
  node [
    id 5
    label "iot_B3"
    memory 536
    ip "172.31.22.90"
  ]
  node [
    id 6
    label "edge_A"
    memory 2147
    ip "172.31.21.155"
  ]
  node [
    id 7
    label "edge_B"
    memory 4294
    ip "172.31.20.197"
  ]
  node [
    id 8
    label "cloud"
    memory 17179
    ip "172.31.22.227"
  ]
  edge [
    source 0
    target 1
    weight [
      bandwidth 200
      delay 8
    ]
  ]
  edge [
    source 0
    target 2
    weight [
      bandwidth 200
      delay 8
    ]
  ]
  edge [
    source 0
    target 3
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 0
    target 4
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 0
    target 5
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 0
    target 6
    weight [
      bandwidth 150
      delay 11
    ]
  ]
  edge [
    source 0
    target 7
    weight [
      bandwidth 100
      delay 31
    ]
  ]
  edge [
    source 0
    target 8
    weight [
      bandwidth 100
      delay 61
    ]
  ]
  edge [
    source 1
    target 2
    weight [
      bandwidth 200
      delay 8
    ]
  ]
  edge [
    source 1
    target 3
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 1
    target 4
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 1
    target 5
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 1
    target 6
    weight [
      bandwidth 150
      delay 11
    ]
  ]
  edge [
    source 1
    target 7
    weight [
      bandwidth 100
      delay 31
    ]
  ]
  edge [
    source 1
    target 8
    weight [
      bandwidth 100
      delay 61
    ]
  ]
  edge [
    source 2
    target 3
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 2
    target 4
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 2
    target 5
    weight [
      bandwidth 15
      delay 51
    ]
  ]
  edge [
    source 2
    target 6
    weight [
      bandwidth 150
      delay 11
    ]
  ]
  edge [
    source 2
    target 7
    weight [
      bandwidth 100
      delay 31
    ]
  ]
  edge [
    source 2
    target 8
    weight [
      bandwidth 100
      delay 61
    ]
  ]
  edge [
    source 3
    target 4
    weight [
      bandwidth 20
      delay 15
    ]
  ]
  edge [
    source 3
    target 5
    weight [
      bandwidth 20
      delay 15
    ]
  ]
  edge [
    source 3
    target 6
    weight [
      bandwidth 15
      delay 40
    ]
  ]
  edge [
    source 3
    target 7
    weight [
      bandwidth 15
      delay 20
    ]
  ]
  edge [
    source 3
    target 8
    weight [
      bandwidth 15
      delay 70
    ]
  ]
  edge [
    source 4
    target 5
    weight [
      bandwidth 20
      delay 15
    ]
  ]
  edge [
    source 4
    target 6
    weight [
      bandwidth 15
      delay 40
    ]
  ]
  edge [
    source 4
    target 7
    weight [
      bandwidth 15
      delay 20
    ]
  ]
  edge [
    source 4
    target 8
    weight [
      bandwidth 15
      delay 70
    ]
  ]
  edge [
    source 5
    target 6
    weight [
      bandwidth 15
      delay 40
    ]
  ]
  edge [
    source 5
    target 7
    weight [
      bandwidth 15
      delay 20
    ]
  ]
  edge [
    source 5
    target 8
    weight [
      bandwidth 15
      delay 70
    ]
  ]
  edge [
    source 6
    target 7
    weight [
      bandwidth 100
      delay 20
    ]
  ]
  edge [
    source 6
    target 8
    weight [
      bandwidth 100
      delay 50
    ]
  ]
  edge [
    source 7
    target 8
    weight [
      bandwidth 100
      delay 50
    ]
  ]
]
