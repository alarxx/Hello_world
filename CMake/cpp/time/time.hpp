/*
    SPDX-License-Identifier: MPL-2.0
    --------------------------------
    This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
    If a copy of the MPL was not distributed with this file,
    You can obtain one at https://mozilla.org/MPL/2.0/.

    Provided “as is”, without warranty of any kind.

    Copyright © 2025 Alar Akilbekov. All rights reserved.
 */

#pragma once
#ifndef _TIME_HPP_
#define _TIME_HPP_

#include <chrono> // vs. ctime
#include <cstdint> // std::intmax_t
#include <ratio> // std::nano

// nanoSeconds - Java style name, also could be `get_ns_since_epoch`
std::intmax_t nanoSeconds(){
    // time_point <typename Clock, typename Duration = typename Clock::duration>
    std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
    // duration < Representation (int or double), Period = std::ratio<1, 1> >
    // Epoch - Duration from (1970-01-01)
    // time_since_epoch возвращает время в nanoseconds
    // chrono::nanoseconds : chrono::duration<int, std::nano>, nano - std::ratio<1, 1*10^9> i.e. 1/10^9
    std::chrono::duration<std::intmax_t, std::nano> epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
    // Workflow:
    // Clock -> now -> time_point -> time_since_epoch -> duration -> count -> rep (number of ticks)
}

#endif
