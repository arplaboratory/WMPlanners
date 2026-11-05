cmake_minimum_required(VERSION 3.18)
project(wm_planner_ros VERSION 0.1.0 LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

if(AMENT_CMAKE_SYMLINK_INSTALL)
endif()

find_package(ament_cmake QUIET)

if(ament_cmake_FOUND)
  find_package(ament_cmake_python REQUIRED)

  install(FILES package.xml DESTINATION share/${PROJECT_NAME})
  install(FILES pyproject.toml DESTINATION share/${PROJECT_NAME})
  install(DIRECTORY resource/ DESTINATION share/${PROJECT_NAME}/resource)
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch
    PATTERN "__pycache__" EXCLUDE)
  install(DIRECTORY configs/ DESTINATION share/${PROJECT_NAME}/configs)
  install(DIRECTORY ../../configs/robots/ DESTINATION share/${PROJECT_NAME}/configs/robots
    FILES_MATCHING PATTERN "*.yaml")

  install(PROGRAMS python/wm_planner_ros/planner_bridge_node.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME wm_planner_bridge_node
  )

  install(PROGRAMS python/wm_planner_ros/dino_wm_planner_bridge_node.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME dino_wm_planner_bridge_node
  )

  ament_python_install_package(wm_planner_ros PACKAGE_DIR python/wm_planner_ros)
  ament_package()
endif()
