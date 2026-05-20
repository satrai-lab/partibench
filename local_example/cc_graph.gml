graph [
  node [
    id 0
    label "node_A"
    memory 99999
    ip "127.0.0.1"
  ]
  node [
    id 1
    label "node_B"
    memory 99999
    ip "127.0.0.2"
  ]
  edge [
    source 0
    target 1
    weight [
      bandwidth 10000
      delay 0
    ]
  ]
]
