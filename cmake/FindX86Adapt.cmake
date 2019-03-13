# Copyright (c) 2016, Technische Universität Dresden, Germany
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include(${CMAKE_CURRENT_LIST_DIR}/UnsetIfUpdated.cmake)

option(X86Adapt_STATIC "Link x86_adapt library static." ON)

UnsetIfUpdated(X86_ADAPT_LIBRARIES X86Adapt_STATIC)

if(X86Adapt_STATIC)
    set(LIBX86A_NAME libx86_adapt_static.a)
else()
    set(LIBX86A_NAME libx86_adapt.so)
endif()

if (X86_ADAPT_LIBRARIES AND X86_ADAPT_INCLUDE_DIRS)
    set (X86Adapt_FIND_QUIETLY TRUE)
endif (X86_ADAPT_LIBRARIES AND X86_ADAPT_INCLUDE_DIRS)

find_path(X86_ADAPT_INCLUDE_DIRS x86_adapt.h HINTS ${X86_ADAPT_DIR} PATHS ENV C_INCLUDE_PATH PATH_SUFFIXES include)
find_library(X86_ADAPT_LIBRARIES NAMES ${LIBX86A_NAME} PATHS ENV LIBRARY_PATH LD_LIBRARY_PATH)

include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(X86Adapt DEFAULT_MSG
        X86_ADAPT_LIBRARIES
        X86_ADAPT_INCLUDE_DIRS)

if (X86Adapt_FOUND)
    add_library(X86Adapt::X86Adapt UNKNOWN IMPORTED GLOBAL)
    set_target_properties(X86Adapt::X86Adapt PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${X86_ADAPT_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${X86_ADAPT_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(X86_ADAPT_INCLUDE_DIRS X86_ADAPT_LIBRARIES)

unset(LIBX86A_NAME)
